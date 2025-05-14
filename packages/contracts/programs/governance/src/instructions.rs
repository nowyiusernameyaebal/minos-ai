//! Instructions for the governance program
//!
//! This module defines the instructions for the governance program, which allows
//! token holders to create, vote on, and execute proposals for protocol changes.
//! The governance module supports various voting mechanisms, delegation, and timelock
//! functionality to ensure secure and transparent governance.

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use solana_program::{program_error::ProgramError, system_program};

use crate::errors::GovernanceError;
use crate::proposals::{Proposal, ProposalAction, ProposalState, ProposalType, VoteOption, VoteWeight};
use crate::state::{
    DelegateRecord, GovernanceConfig, GovernanceTreasury, ProposalStatistics, TokenLock, 
    VoteRecord, Voter, GOVERNANCE_PDA_SEED, PROPOSAL_PDA_SEED, TREASURY_PDA_SEED, 
    VOTE_RECORD_PDA_SEED, DELEGATE_PDA_SEED, TOKEN_LOCK_PDA_SEED,
};

/// The maximum length for proposal titles
pub const MAX_TITLE_LENGTH: usize = 100;

/// The maximum length for proposal descriptions
pub const MAX_DESCRIPTION_LENGTH: usize = 2048;

/// The maximum number of actions per proposal
pub const MAX_ACTIONS_PER_PROPOSAL: usize = 10;

/// The minimum amount of tokens required to create a proposal
pub const MIN_PROPOSAL_THRESHOLD: u64 = 100_000_000; // 100 tokens with 6 decimals

/// The minimum voting period in seconds (3 days)
pub const MIN_VOTING_PERIOD: i64 = 259_200;

/// The maximum voting period in seconds (14 days)
pub const MAX_VOTING_PERIOD: i64 = 1_209_600;

/// The minimum timelock period in seconds (24 hours)
pub const MIN_TIMELOCK_PERIOD: i64 = 86_400;

/// The maximum timelock period in seconds (30 days)
pub const MAX_TIMELOCK_PERIOD: i64 = 2_592_000;

/// Initialize the governance program with the specified configuration
#[derive(Accounts)]
#[instruction(config: GovernanceConfig)]
pub struct Initialize<'info> {
    /// The authority that can initialize the governance program
    #[account(mut)]
    pub authority: Signer<'info>,

    /// The governance state account
    #[account(
        init,
        payer = authority,
        space = 8 + std::mem::size_of::<GovernanceConfig>(),
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The treasury account that will hold governance funds
    #[account(
        init,
        payer = authority,
        space = 8 + std::mem::size_of::<GovernanceTreasury>(),
        seeds = [TREASURY_PDA_SEED],
        bump,
    )]
    pub treasury: Account<'info, GovernanceTreasury>,

    /// The governance token mint
    pub governance_token_mint: AccountInfo<'info>,

    /// The system program
    pub system_program: Program<'info, System>,

    /// The rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Update the governance configuration
#[derive(Accounts)]
pub struct UpdateConfig<'info> {
    /// The authority that can update the governance program
    pub authority: Signer<'info>,

    /// The governance state account
    #[account(
        mut,
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
        constraint = governance.authority == *authority.key @ GovernanceError::UnauthorizedAccess
    )]
    pub governance: Account<'info, GovernanceConfig>,
}

/// Create a new proposal
#[derive(Accounts)]
#[instruction(
    proposal_type: ProposalType,
    title: String,
    description: String,
    voting_period: i64,
    execution_delay: i64,
    proposal_index: u64,
)]
pub struct CreateProposal<'info> {
    /// The user creating the proposal
    #[account(mut)]
    pub proposer: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        init,
        payer = proposer,
        space = 8 + std::mem::size_of::<Proposal>() + title.len() + description.len(),
        seeds = [PROPOSAL_PDA_SEED, &proposal_index.to_le_bytes()],
        bump,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The token account of the proposer, which will be used to verify the proposal threshold
    #[account(
        constraint = proposer_token_account.owner == *proposer.key @ GovernanceError::InvalidOwner,
        constraint = proposer_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
    )]
    pub proposer_token_account: Account<'info, TokenAccount>,

    /// The system program
    pub system_program: Program<'info, System>,

    /// The rent sysvar
    pub rent: Sysvar<'info, Rent>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Add an action to a proposal
#[derive(Accounts)]
#[instruction(action: ProposalAction)]
pub struct AddProposalAction<'info> {
    /// The user adding the action
    pub proposer: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = proposal.proposer == *proposer.key @ GovernanceError::UnauthorizedAccess,
        constraint = proposal.state == ProposalState::Draft @ GovernanceError::InvalidProposalState,
        constraint = proposal.actions.len() < MAX_ACTIONS_PER_PROPOSAL @ GovernanceError::TooManyActions,
    )]
    pub proposal: Account<'info, Proposal>,
}

/// Cancel a proposal
#[derive(Accounts)]
pub struct CancelProposal<'info> {
    /// The user cancelling the proposal
    pub authority: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = (proposal.proposer == *authority.key || governance.authority == *authority.key) @ GovernanceError::UnauthorizedAccess,
        constraint = proposal.state == ProposalState::Draft || proposal.state == ProposalState::Active @ GovernanceError::InvalidProposalState,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Activate a proposal for voting
#[derive(Accounts)]
pub struct ActivateProposal<'info> {
    /// The proposer activating the proposal
    pub proposer: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = proposal.proposer == *proposer.key @ GovernanceError::UnauthorizedAccess,
        constraint = proposal.state == ProposalState::Draft @ GovernanceError::InvalidProposalState,
        constraint = !proposal.actions.is_empty() @ GovernanceError::NoProposalActions,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The token account of the proposer, which will be used to verify the proposal threshold
    #[account(
        constraint = proposer_token_account.owner == *proposer.key @ GovernanceError::InvalidOwner,
        constraint = proposer_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
        constraint = proposer_token_account.amount >= MIN_PROPOSAL_THRESHOLD @ GovernanceError::InsufficientTokens,
    )]
    pub proposer_token_account: Account<'info, TokenAccount>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Cast a vote on a proposal
#[derive(Accounts)]
#[instruction(vote: VoteOption)]
pub struct CastVote<'info> {
    /// The voter casting the vote
    #[account(mut)]
    pub voter: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = proposal.state == ProposalState::Active @ GovernanceError::InvalidProposalState,
        constraint = Clock::get()?.unix_timestamp <= proposal.voting_ends_at @ GovernanceError::VotingPeriodEnded,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The vote record account
    #[account(
        init_if_needed,
        payer = voter,
        space = 8 + std::mem::size_of::<VoteRecord>(),
        seeds = [VOTE_RECORD_PDA_SEED, proposal.key().as_ref(), voter.key().as_ref()],
        bump,
    )]
    pub vote_record: Account<'info, VoteRecord>,

    /// The token account of the voter
    #[account(
        constraint = voter_token_account.owner == *voter.key @ GovernanceError::InvalidOwner,
        constraint = voter_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
    )]
    pub voter_token_account: Account<'info, TokenAccount>,

    /// Optional token lock account to increase voting power
    #[account(
        mut,
        seeds = [TOKEN_LOCK_PDA_SEED, voter.key().as_ref()],
        bump,
        constraint = token_lock.owner == *voter.key @ GovernanceError::InvalidOwner,
    )]
    pub token_lock: Option<Account<'info, TokenLock>>,

    /// The system program
    pub system_program: Program<'info, System>,

    /// The rent sysvar
    pub rent: Sysvar<'info, Rent>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Create a delegate for voting
#[derive(Accounts)]
pub struct CreateDelegate<'info> {
    /// The voter delegating their voting power
    #[account(mut)]
    pub voter: Signer<'info>,

    /// The delegate being authorized
    /// CHECK: This is just a public key, not an account we're reading from
    pub delegate: AccountInfo<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The delegate record account
    #[account(
        init,
        payer = voter,
        space = 8 + std::mem::size_of::<DelegateRecord>(),
        seeds = [DELEGATE_PDA_SEED, voter.key().as_ref(), delegate.key().as_ref()],
        bump,
    )]
    pub delegate_record: Account<'info, DelegateRecord>,

    /// The system program
    pub system_program: Program<'info, System>,

    /// The rent sysvar
    pub rent: Sysvar<'info, Rent>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Remove a delegate
#[derive(Accounts)]
pub struct RemoveDelegate<'info> {
    /// The voter removing their delegate
    #[account(mut)]
    pub voter: Signer<'info>,

    /// The delegate being removed
    /// CHECK: This is just a public key, not an account we're reading from
    pub delegate: AccountInfo<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The delegate record account
    #[account(
        mut,
        close = voter,
        seeds = [DELEGATE_PDA_SEED, voter.key().as_ref(), delegate.key().as_ref()],
        bump,
        constraint = delegate_record.voter == *voter.key @ GovernanceError::UnauthorizedAccess,
        constraint = delegate_record.delegate == *delegate.key @ GovernanceError::InvalidDelegate,
    )]
    pub delegate_record: Account<'info, DelegateRecord>,
}

/// Lock tokens for increased voting power
#[derive(Accounts)]
#[instruction(amount: u64, lock_duration: i64)]
pub struct LockTokens<'info> {
    /// The voter locking their tokens
    #[account(mut)]
    pub voter: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The token lock account
    #[account(
        init_if_needed,
        payer = voter,
        space = 8 + std::mem::size_of::<TokenLock>(),
        seeds = [TOKEN_LOCK_PDA_SEED, voter.key().as_ref()],
        bump,
    )]
    pub token_lock: Account<'info, TokenLock>,

    /// The token account of the voter
    #[account(
        mut,
        constraint = voter_token_account.owner == *voter.key @ GovernanceError::InvalidOwner,
        constraint = voter_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
        constraint = voter_token_account.amount >= amount @ GovernanceError::InsufficientTokens,
    )]
    pub voter_token_account: Account<'info, TokenAccount>,

    /// The token account that will hold the locked tokens
    #[account(
        mut,
        constraint = escrow_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    /// The token program
    pub token_program: Program<'info, Token>,

    /// The system program
    pub system_program: Program<'info, System>,

    /// The rent sysvar
    pub rent: Sysvar<'info, Rent>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Unlock tokens after the lock period has expired
#[derive(Accounts)]
pub struct UnlockTokens<'info> {
    /// The voter unlocking their tokens
    #[account(mut)]
    pub voter: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The token lock account
    #[account(
        mut,
        seeds = [TOKEN_LOCK_PDA_SEED, voter.key().as_ref()],
        bump,
        constraint = token_lock.owner == *voter.key @ GovernanceError::InvalidOwner,
        constraint = Clock::get()?.unix_timestamp >= token_lock.unlock_timestamp @ GovernanceError::TokensStillLocked,
    )]
    pub token_lock: Account<'info, TokenLock>,

    /// The token account that will receive the unlocked tokens
    #[account(
        mut,
        constraint = voter_token_account.owner == *voter.key @ GovernanceError::InvalidOwner,
        constraint = voter_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
    )]
    pub voter_token_account: Account<'info, TokenAccount>,

    /// The token account that holds the locked tokens
    #[account(
        mut,
        constraint = escrow_token_account.mint == governance.governance_token_mint @ GovernanceError::InvalidTokenMint,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    /// The token program
    pub token_program: Program<'info, Token>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Finalize a proposal after the voting period has ended
#[derive(Accounts)]
pub struct FinalizeProposal<'info> {
    /// The user finalizing the proposal
    pub user: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = proposal.state == ProposalState::Active @ GovernanceError::InvalidProposalState,
        constraint = Clock::get()?.unix_timestamp > proposal.voting_ends_at @ GovernanceError::VotingPeriodNotEnded,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Execute a successful proposal after the timelock period
#[derive(Accounts)]
pub struct ExecuteProposal<'info> {
    /// The user executing the proposal
    pub executor: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        mut,
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
        constraint = proposal.state == ProposalState::Succeeded @ GovernanceError::InvalidProposalState,
        constraint = Clock::get()?.unix_timestamp >= proposal.execution_timestamp @ GovernanceError::TimelockNotExpired,
    )]
    pub proposal: Account<'info, Proposal>,

    /// The treasury account that may be required for fund transfers
    #[account(
        mut,
        seeds = [TREASURY_PDA_SEED],
        bump,
    )]
    pub treasury: Account<'info, GovernanceTreasury>,
    
    /// The system program
    pub system_program: Program<'info, System>,

    /// The token program (for token transfers)
    pub token_program: Program<'info, Token>,

    /// The clock sysvar
    pub clock: Sysvar<'info, Clock>,
}

/// Deposit funds into the governance treasury
#[derive(Accounts)]
#[instruction(amount: u64)]
pub struct DepositTreasury<'info> {
    /// The user depositing funds
    #[account(mut)]
    pub depositor: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The treasury account
    #[account(
        mut,
        seeds = [TREASURY_PDA_SEED],
        bump,
    )]
    pub treasury: Account<'info, GovernanceTreasury>,

    /// The token account of the depositor
    #[account(
        mut,
        constraint = depositor_token_account.owner == *depositor.key @ GovernanceError::InvalidOwner,
    )]
    pub depositor_token_account: Account<'info, TokenAccount>,

    /// The token account of the treasury
    #[account(mut)]
    pub treasury_token_account: Account<'info, TokenAccount>,

    /// The token program
    pub token_program: Program<'info, Token>,
}

/// Withdraw funds from the governance treasury (requires a successful proposal)
#[derive(Accounts)]
#[instruction(amount: u64)]
pub struct WithdrawTreasury<'info> {
    /// The user withdrawing funds (must be authorized via a successful proposal)
    #[account(mut)]
    pub recipient: Signer<'info>,

    /// The governance state account
    #[account(
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The treasury account
    #[account(
        mut,
        seeds = [TREASURY_PDA_SEED],
        bump,
    )]
    pub treasury: Account<'info, GovernanceTreasury>,

    /// The proposal that authorized this withdrawal
    #[account(
        constraint = proposal.state == ProposalState::Executed @ GovernanceError::InvalidProposalState,
        // Additional verification of withdrawal authorization would be implemented in the actual execution
    )]
    pub proposal: Account<'info, Proposal>,

    /// The token account of the recipient
    #[account(
        mut,
        constraint = recipient_token_account.owner == *recipient.key @ GovernanceError::InvalidOwner,
    )]
    pub recipient_token_account: Account<'info, TokenAccount>,

    /// The token account of the treasury
    #[account(
        mut,
        constraint = treasury_token_account.amount >= amount @ GovernanceError::InsufficientTreasuryFunds,
    )]
    pub treasury_token_account: Account<'info, TokenAccount>,

    /// The token program
    pub token_program: Program<'info, Token>,
}

/// Update proposal statistics after voting or finalization
#[derive(Accounts)]
pub struct UpdateProposalStats<'info> {
    /// The governance state account
    #[account(
        mut,
        seeds = [GOVERNANCE_PDA_SEED],
        bump,
    )]
    pub governance: Account<'info, GovernanceConfig>,

    /// The proposal account
    #[account(
        seeds = [PROPOSAL_PDA_SEED, &proposal.index.to_le_bytes()],
        bump,
    )]
    pub proposal: Account<'info, Proposal>,
}

/// Initialize the instruction handler
pub fn initialize(ctx: Context<Initialize>, config: GovernanceConfig) -> Result<()> {
    let governance = &mut ctx.accounts.governance;
    let treasury = &mut ctx.accounts.treasury;
    
    // Set the governance configuration
    governance.authority = *ctx.accounts.authority.key;
    governance.governance_token_mint = *ctx.accounts.governance_token_mint.key;
    governance.proposal_count = 0;
    governance.quorum_percentage = config.quorum_percentage;
    governance.approval_threshold = config.approval_threshold;
    governance.voting_delay = config.voting_delay;
    governance.voting_period = config.voting_period;
    governance.timelock_delay = config.timelock_delay;
    
    // Validate configuration parameters
    require!(
        governance.quorum_percentage > 0 && governance.quorum_percentage <= 100,
        GovernanceError::InvalidQuorumPercentage
    );
    require!(
        governance.approval_threshold > 0 && governance.approval_threshold <= 100,
        GovernanceError::InvalidApprovalThreshold
    );
    require!(
        governance.voting_period >= MIN_VOTING_PERIOD && governance.voting_period <= MAX_VOTING_PERIOD,
        GovernanceError::InvalidVotingPeriod
    );
    require!(
        governance.timelock_delay >= MIN_TIMELOCK_PERIOD && governance.timelock_delay <= MAX_TIMELOCK_PERIOD,
        GovernanceError::InvalidTimelockPeriod
    );
    
    // Initialize treasury
    treasury.authority = governance.key();
    treasury.proposal_count = 0;
    treasury.total_deposits = 0;
    treasury.total_withdrawals = 0;
    
    msg!("Governance initialized with token mint: {}", governance.governance_token_mint);
    
    Ok(())
}

/// Create a delegate for voting
pub fn create_delegate(ctx: Context<CreateDelegate>) -> Result<()> {
    let delegate_record = &mut ctx.accounts.delegate_record;
    let clock = &ctx.accounts.clock;
    
    // Initialize the delegate record
    delegate_record.voter = ctx.accounts.voter.key();
    delegate_record.delegate = ctx.accounts.delegate.key();
    delegate_record.created_at = clock.unix_timestamp;
    delegate_record.active = true;
    
    msg!("Delegate created for voter {}", delegate_record.voter);
    
    Ok(())
}

/// Remove a delegate
pub fn remove_delegate(ctx: Context<RemoveDelegate>) -> Result<()> {
    // No need to update the delegate record as it will be closed
    msg!("Delegate removed for voter {}", ctx.accounts.voter.key());
    
    Ok(())
}

/// Lock tokens for increased voting power
pub fn lock_tokens(ctx: Context<LockTokens>, amount: u64, lock_duration: i64) -> Result<()> {
    let token_lock = &mut ctx.accounts.token_lock;
    let voter = &ctx.accounts.voter;
    let clock = &ctx.accounts.clock;
    
    // Validate lock duration
    require!(
        lock_duration >= MIN_TIMELOCK_PERIOD && lock_duration <= 365 * 86400, // Maximum 1 year
        GovernanceError::InvalidLockDuration
    );
    
    // Calculate unlock timestamp
    let unlock_timestamp = clock.unix_timestamp.saturating_add(lock_duration);
    
    // Initialize token lock if new
    if token_lock.owner.is_empty() {
        token_lock.owner = *voter.key;
        token_lock.created_at = clock.unix_timestamp;
    }
    
    // Update token lock
    token_lock.locked_amount = token_lock.locked_amount.saturating_add(amount);
    token_lock.unlock_timestamp = std::cmp::max(token_lock.unlock_timestamp, unlock_timestamp);
    
    // Transfer tokens to escrow
    let transfer_instruction = Transfer {
        from: ctx.accounts.voter_token_account.to_account_info(),
        to: ctx.accounts.escrow_token_account.to_account_info(),
        authority: voter.to_account_info(),
    };
    
    let cpi_ctx = CpiContext::new(
        ctx.accounts.token_program.to_account_info(),
        transfer_instruction,
    );
    
    token::transfer(cpi_ctx, amount)?;
    
    msg!("Locked {} tokens until {}", amount, unlock_timestamp);
    
    Ok(())
}

/// Unlock tokens after the lock period has expired
pub fn unlock_tokens(ctx: Context<UnlockTokens>) -> Result<()> {
    let token_lock = &mut ctx.accounts.token_lock;
    let amount = token_lock.locked_amount;
    
    // Reset token lock
    token_lock.locked_amount = 0;
    
    // Transfer tokens from escrow back to voter
    let escrow_bump = *ctx.bumps.get("token_lock").unwrap();
    let escrow_seeds = &[TOKEN_LOCK_PDA_SEED, ctx.accounts.voter.key().as_ref(), &[escrow_bump]];
    let signer_seeds = &[&escrow_seeds[..]];
    
    let transfer_instruction = Transfer {
        from: ctx.accounts.escrow_token_account.to_account_info(),
        to: ctx.accounts.voter_token_account.to_account_info(),
        authority: token_lock.to_account_info(),
    };
    
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        transfer_instruction,
        signer_seeds,
    );
    
    token::transfer(cpi_ctx, amount)?;
    
    msg!("Unlocked {} tokens", amount);
    
    Ok(())
}

/// Finalize a proposal after the voting period has ended
pub fn finalize_proposal(ctx: Context<FinalizeProposal>) -> Result<()> {
    let governance = &ctx.accounts.governance;
    let proposal = &mut ctx.accounts.proposal;
    let clock = &ctx.accounts.clock;
    
    // Calculate voting results
    let total_votes = proposal.yes_votes.saturating_add(proposal.no_votes);
    
    // Calculate quorum percentage (considering only yes and no votes)
    let governance_token_supply = 1_000_000_000_000; // This should be fetched from the token mint
    let quorum_threshold = (governance_token_supply * governance.quorum_percentage as u64) / 100;
    
    // Check if quorum is reached
    let quorum_reached = total_votes >= quorum_threshold;
    
    // Calculate approval percentage
    let approval_percentage = if total_votes > 0 {
        (proposal.yes_votes * 100) / total_votes
    } else {
        0
    };
    
    // Determine the proposal outcome
    if !quorum_reached {
        proposal.state = ProposalState::Defeated;
        msg!("Proposal {} defeated: Quorum not reached", proposal.index);
    } else if approval_percentage >= governance.approval_threshold {
        proposal.state = ProposalState::Succeeded;
        
        // Set the execution timestamp with timelock delay
        proposal.execution_timestamp = clock.unix_timestamp.saturating_add(proposal.execution_delay);
        
        msg!("Proposal {} succeeded with {}% approval", proposal.index, approval_percentage);
        msg!("Execution will be available at timestamp {}", proposal.execution_timestamp);
    } else {
        proposal.state = ProposalState::Defeated;
        msg!("Proposal {} defeated: Approval threshold not met", proposal.index);
    }
    
    Ok(())
}

/// Execute a successful proposal after the timelock period
pub fn execute_proposal(ctx: Context<ExecuteProposal>) -> Result<()> {
    let proposal = &mut ctx.accounts.proposal;
    
    // Mark the proposal as executed
    proposal.state = ProposalState::Executed;
    
    // Note: In a real implementation, this would execute the proposal's actions
    // which could involve various CPI calls to other programs based on the
    // actions specified in the proposal.
    
    msg!("Proposal {} executed", proposal.index);
    
    Ok(())
}

/// Deposit funds into the governance treasury
pub fn deposit_treasury(ctx: Context<DepositTreasury>, amount: u64) -> Result<()> {
    let treasury = &mut ctx.accounts.treasury;
    
    // Transfer tokens to the treasury
    let transfer_instruction = Transfer {
        from: ctx.accounts.depositor_token_account.to_account_info(),
        to: ctx.accounts.treasury_token_account.to_account_info(),
        authority: ctx.accounts.depositor.to_account_info(),
    };
    
    let cpi_ctx = CpiContext::new(
        ctx.accounts.token_program.to_account_info(),
        transfer_instruction,
    );
    
    token::transfer(cpi_ctx, amount)?;
    
    // Update treasury stats
    treasury.total_deposits = treasury.total_deposits.saturating_add(amount);
    
    msg!("Deposited {} tokens to treasury", amount);
    
    Ok(())
}

/// Withdraw funds from the governance treasury (requires a successful proposal)
pub fn withdraw_treasury(ctx: Context<WithdrawTreasury>, amount: u64) -> Result<()> {
    let treasury = &mut ctx.accounts.treasury;
    let governance = &ctx.accounts.governance;
    
    // In a full implementation, there would be verification that this withdrawal
    // was authorized by the specified proposal
    
    // Transfer tokens from the treasury
    let treasury_bump = *ctx.bumps.get("treasury").unwrap();
    let treasury_seeds = &[TREASURY_PDA_SEED, &[treasury_bump]];
    let signer_seeds = &[&treasury_seeds[..]];
    
    let transfer_instruction = Transfer {
        from: ctx.accounts.treasury_token_account.to_account_info(),
        to: ctx.accounts.recipient_token_account.to_account_info(),
        authority: treasury.to_account_info(),
    };
    
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        transfer_instruction,
        signer_seeds,
    );
    
    token::transfer(cpi_ctx, amount)?;
    
    // Update treasury stats
    treasury.total_withdrawals = treasury.total_withdrawals.saturating_add(amount);
    
    msg!("Withdrew {} tokens from treasury", amount);
    
    Ok(())
}

/// Update proposal statistics after voting or finalization
pub fn update_proposal_stats(ctx: Context<UpdateProposalStats>) -> Result<()> {
    let governance = &mut ctx.accounts.governance;
    let proposal = &ctx.accounts.proposal;
    
    // Update governance stats based on proposal state
    match proposal.state {
        ProposalState::Executed => {
            governance.executed_proposal_count = governance.executed_proposal_count.saturating_add(1);
        },
        ProposalState::Defeated => {
            governance.defeated_proposal_count = governance.defeated_proposal_count.saturating_add(1);
        },
        _ => {}
    }
    
    msg!("Updated governance statistics");
    
    Ok(())
}

/// Update the governance configuration
pub fn update_config(ctx: Context<UpdateConfig>, config: GovernanceConfig) -> Result<()> {
    let governance = &mut ctx.accounts.governance;
    
    // Update the governance configuration parameters that are allowed to be changed
    governance.quorum_percentage = config.quorum_percentage;
    governance.approval_threshold = config.approval_threshold;
    governance.voting_delay = config.voting_delay;
    governance.voting_period = config.voting_period;
    governance.timelock_delay = config.timelock_delay;
    
    // Validate configuration parameters
    require!(
        governance.quorum_percentage > 0 && governance.quorum_percentage <= 100,
        GovernanceError::InvalidQuorumPercentage
    );
    require!(
        governance.approval_threshold > 0 && governance.approval_threshold <= 100,
        GovernanceError::InvalidApprovalThreshold
    );
    require!(
        governance.voting_period >= MIN_VOTING_PERIOD && governance.voting_period <= MAX_VOTING_PERIOD,
        GovernanceError::InvalidVotingPeriod
    );
    require!(
        governance.timelock_delay >= MIN_TIMELOCK_PERIOD && governance.timelock_delay <= MAX_TIMELOCK_PERIOD,
        GovernanceError::InvalidTimelockPeriod
    );
    
    msg!("Governance configuration updated");
    
    Ok(())
}

/// Create a new proposal
pub fn create_proposal(
    ctx: Context<CreateProposal>,
    proposal_type: ProposalType,
    title: String,
    description: String,
    voting_period: i64,
    execution_delay: i64,
    proposal_index: u64,
) -> Result<()> {
    let governance = &ctx.accounts.governance;
    let proposal = &mut ctx.accounts.proposal;
    let proposer = &ctx.accounts.proposer;
    let clock = &ctx.accounts.clock;
    
    // Validate proposal parameters
    require!(
        title.len() <= MAX_TITLE_LENGTH,
        GovernanceError::TitleTooLong
    );
    require!(
        description.len() <= MAX_DESCRIPTION_LENGTH,
        GovernanceError::DescriptionTooLong
    );
    require!(
        voting_period >= MIN_VOTING_PERIOD && voting_period <= MAX_VOTING_PERIOD,
        GovernanceError::InvalidVotingPeriod
    );
    require!(
        execution_delay >= MIN_TIMELOCK_PERIOD && execution_delay <= MAX_TIMELOCK_PERIOD,
        GovernanceError::InvalidTimelockPeriod
    );
    
    // Initialize the proposal
    proposal.proposer = *proposer.key;
    proposal.index = proposal_index;
    proposal.proposal_type = proposal_type;
    proposal.title = title;
    proposal.description = description;
    proposal.state = ProposalState::Draft;
    proposal.created_at = clock.unix_timestamp;
    proposal.voting_starts_at = 0; // Will be set upon activation
    proposal.voting_ends_at = 0; // Will be set upon activation
    proposal.execution_timestamp = 0; // Will be set upon finalization
    proposal.yes_votes = 0;
    proposal.no_votes = 0;
    proposal.abstain_votes = 0;
    proposal.total_voting_power = 0;
    proposal.voting_period = voting_period;
    proposal.execution_delay = execution_delay;
    proposal.actions = Vec::new();
    
    msg!("Proposal created with index: {}", proposal.index);
    
    Ok(())
}

/// Add an action to a proposal
pub fn add_proposal_action(ctx: Context<AddProposalAction>, action: ProposalAction) -> Result<()> {
    let proposal = &mut ctx.accounts.proposal;
    
    // Add the action to the proposal
    proposal.actions.push(action);
    
    msg!("Action added to proposal {}", proposal.index);
    
    Ok(())
}

/// Cancel a proposal
pub fn cancel_proposal(ctx: Context<CancelProposal>) -> Result<()> {
    let proposal = &mut ctx.accounts.proposal;
    
    // Update the proposal state
    proposal.state = ProposalState::Cancelled;
    
    msg!("Proposal {} cancelled", proposal.index);
    
    Ok(())
}

/// Activate a proposal for voting
pub fn activate_proposal(ctx: Context<ActivateProposal>) -> Result<()> {
    let governance = &ctx.accounts.governance;
    let proposal = &mut ctx.accounts.proposal;
    let clock = &ctx.accounts.clock;
    
    // Calculate voting period timestamps
    let current_time = clock.unix_timestamp;
    let voting_starts_at = current_time + governance.voting_delay;
    let voting_ends_at = voting_starts_at + proposal.voting_period;
    
    // Update the proposal state
    proposal.state = ProposalState::Active;
    proposal.voting_starts_at = voting_starts_at;
    proposal.voting_ends_at = voting_ends_at;
    
    msg!("Proposal {} activated for voting", proposal.index);
    msg!("Voting period: {} to {}", voting_starts_at, voting_ends_at);
    
    Ok(())
}

/// Cast a vote on a proposal
/// Cast a vote on a proposal
pub fn cast_vote(ctx: Context<CastVote>, vote: VoteOption) -> Result<()> {
    let proposal = &mut ctx.accounts.proposal;
    let vote_record = &mut ctx.accounts.vote_record;
    let voter_token_account = &ctx.accounts.voter_token_account;
    let token_lock = &ctx.accounts.token_lock;
    let clock = &ctx.accounts.clock;
    
    // Calculate voting power
    let mut voting_power = voter_token_account.amount;
    
    // Add bonus voting power if tokens are locked
    if let Some(lock) = token_lock {
        let lock_remaining = lock.unlock_timestamp.saturating_sub(clock.unix_timestamp);
        if lock_remaining > 0 {
            // Calculate bonus based on remaining lock time (example formula)
            let max_bonus_multiplier = 2; // Maximum 2x bonus
            let max_lock_duration = 365 * 86400; // 1 year in seconds
            let bonus_multiplier = 1.0 + ((lock_remaining as f64 / max_lock_duration as f64) * (max_bonus_multiplier - 1) as f64);
            voting_power = (voting_power as f64 * bonus_multiplier) as u64;
        }
    }
    
    // Check if the voter has already voted
    if vote_record.voted {
        return Err(GovernanceError::AlreadyVoted.into());
    }
    
    // Record the vote
    vote_record.voter = ctx.accounts.voter.key();
    vote_record.proposal = proposal.key();
    vote_record.vote = vote;
    vote_record.voting_power = voting_power;
    vote_record.voted = true;
    vote_record.vote_timestamp = clock.unix_timestamp;
    
    // Update proposal vote counts
    match vote {
        VoteOption::Yes => {
            proposal.yes_votes = proposal.yes_votes.saturating_add(voting_power);
        }
        VoteOption::No => {
            proposal.no_votes = proposal.no_votes.saturating_add(voting_power);
        }
        VoteOption::Abstain => {
            proposal.abstain_votes = proposal.abstain_votes.saturating_add(voting_power);
        }
    }
    
    // Update total voting power
    proposal.total_voting_power = proposal.total_voting_power.saturating_add(voting_power);
    
    msg!("Vote cast on proposal {}: {:?} with power {}", proposal.index, vote, voting_power);
    
    Ok(())