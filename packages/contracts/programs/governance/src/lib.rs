/**
 * @title Minos AI Governance Program
 * @notice Advanced on-chain governance protocol implementing decentralized decision-making for the Minos-AI platform
 * @dev This program enables token-weighted voting, proposal creation and execution, with sophisticated timelock 
 *      and delegation mechanisms to ensure security, transparency, and effective decentralized governance.
 * 
 * Architecture overview:
 * 1. Governance Realms - Top-level containers for governing specific protocols
 * 2. Proposals - On-chain executable proposals with configurable instructions
 * 3. Vote Management - Token-weighted voting with advanced mechanisms (quadratic, delegated, etc.)
 * 4. Execution Layer - Secure cross-program execution of approved governance actions
 * 
 * @custom:security-contact security@minos-ai.com
 * @custom:audited-by Quantstamp, Zellic, OtterSec
 * @author Minos-AI Governance Team
 * @copyright 2025 Minos-AI, all rights reserved
 * @version 1.0.0
 */

use anchor_lang::{
    prelude::*,
    solana_program::{
        program::invoke_signed,
        instruction::{Instruction as SolanaInstruction, AccountMeta},
        program_error::ProgramError,
        program_pack::Pack,
        pubkey::Pubkey,
        system_instruction,
        sysvar::{clock::Clock, rent::Rent},
    },
    AccountDeserialize, AnchorDeserialize, AnchorSerialize, Discriminator,
    InstructionData, Key, ToAccountMetas
};

use anchor_spl::{
    associated_token::AssociatedToken,
    token::{Mint, Token, TokenAccount, Transfer},
};

use borsh::{BorshDeserialize, BorshSerialize};
use std::{
    cmp::{max, min},
    convert::{TryFrom, TryInto},
    mem::size_of,
    str::FromStr,
};

// Program modules
pub mod state;
pub mod instructions;
pub mod proposals;
pub mod errors;
pub mod utils;
pub mod voting;
pub mod security;
pub mod events;
pub mod constants;

// Re-export key types and utilities
pub use state::*;
pub use errors::GovernanceError;
pub use events::*;
pub use constants::*;

// Program ID declaration
declare_id!("GovErdRMLLRQpWX1AQgEYFhdN4mKRVANyTPJKiUeVXkH");

/**
 * Program entrypoint and instruction handlers
 * Implements the governance protocol functionality through a set of instructions
 * that manage governance realms, proposals, voting, and execution
 */
#[program]
pub mod minos_governance {
    use super::*;

    /**
     * Creates a new governance realm
     * 
     * A realm is the top-level governance entity that contains proposals and configurations
     * for a specific protocol or set of programs being governed.
     * 
     * @param ctx The context containing accounts for the realm creation
     * @param name The name of the governance realm (max 32 bytes)
     * @param config Configuration parameters for the realm
     * @return Result indicating success or specific error
     */
    pub fn create_realm(
        ctx: Context<CreateRealm>,
        name: String,
        config: RealmConfig,
    ) -> Result<()> {
        instructions::realm::create_realm(ctx, name, config)
    }

    /**
     * Updates configuration parameters for an existing realm
     * 
     * Only callable by the current realm authority
     * 
     * @param ctx The context containing accounts for the realm update
     * @param config New configuration parameters
     * @return Result indicating success or specific error
     */
    pub fn update_realm_config(
        ctx: Context<UpdateRealmConfig>,
        config: RealmConfig,
    ) -> Result<()> {
        instructions::realm::update_realm_config(ctx, config)
    }

    /**
     * Sets or changes the realm authority
     * 
     * @param ctx The context containing accounts for updating the realm authority
     * @param new_authority New authority public key or None to remove authority
     * @return Result indicating success or specific error
     */
    pub fn set_realm_authority(
        ctx: Context<SetRealmAuthority>,
        new_authority: Option<Pubkey>,
    ) -> Result<()> {
        instructions::realm::set_realm_authority(ctx, new_authority)
    }

    /**
     * Creates a governance token mint with specified parameters
     * 
     * @param ctx The context containing accounts for token creation
     * @param name Token name
     * @param symbol Token symbol
     * @param decimals Number of token decimals
     * @param token_type Type of governance token (standard, membership, etc.)
     * @param min_transfer_amount Minimum transferable amount (for anti-spam)
     * @return Result indicating success or specific error
     */
    pub fn create_governance_token(
        ctx: Context<CreateGovernanceToken>,
        name: String,
        symbol: String,
        decimals: u8,
        token_type: GovernanceTokenType,
        min_transfer_amount: u64,
    ) -> Result<()> {
        instructions::token::create_governance_token(
            ctx, name, symbol, decimals, token_type, min_transfer_amount
        )
    }

    /**
     * Deposits governance tokens to participate in realm governance
     * 
     * @param ctx The context containing accounts for token deposit
     * @param amount Amount of tokens to deposit
     * @return Result indicating success or specific error
     */
    pub fn deposit_governing_tokens(
        ctx: Context<DepositGoverningTokens>,
        amount: u64,
    ) -> Result<()> {
        instructions::token::deposit_governing_tokens(ctx, amount)
    }

    /**
     * Withdraws previously deposited governance tokens
     * 
     * @param ctx The context containing accounts for token withdrawal
     * @param amount Amount of tokens to withdraw
     * @return Result indicating success or specific error
     */
    pub fn withdraw_governing_tokens(
        ctx: Context<WithdrawGoverningTokens>,
        amount: u64,
    ) -> Result<()> {
        instructions::token::withdraw_governing_tokens(ctx, amount)
    }

    /**
     * Creates a new governance proposal
     * 
     * @param ctx The context containing accounts for proposal creation
     * @param title Proposal title
     * @param description Detailed proposal description
     * @param options Vote options for multi-choice proposals
     * @param voting_type Type of voting mechanism
     * @param execution_timelock_seconds Delay between approval and execution
     * @param voting_period_seconds Duration proposal remains open for voting
     * @param vote_threshold_percentage Percentage of votes needed for approval
     * @return Result indicating success or specific error
     */
    pub fn create_proposal(
        ctx: Context<CreateProposal>,
        title: String,
        description: String,
        options: Option<Vec<String>>,
        voting_type: VotingType,
        execution_timelock_seconds: i64,
        voting_period_seconds: i64,
        vote_threshold_percentage: u8,
    ) -> Result<()> {
        instructions::proposal::create_proposal(
            ctx, 
            title, 
            description, 
            options, 
            voting_type,
            execution_timelock_seconds, 
            voting_period_seconds,
            vote_threshold_percentage
        )
    }

    /**
     * Cancels a proposal that hasn't started voting yet
     * 
     * @param ctx The context containing accounts for proposal cancellation
     * @return Result indicating success or specific error
     */
    pub fn cancel_proposal(
        ctx: Context<CancelProposal>,
    ) -> Result<()> {
        instructions::proposal::cancel_proposal(ctx)
    }

    /**
     * Adds an executable instruction to a proposal
     * 
     * @param ctx The context containing accounts for adding instructions
     * @param instruction_index Index position for this instruction
     * @param program_id Program to execute the instruction
     * @param accounts Account metas required for the instruction
     * @param data Instruction data
     * @return Result indicating success or specific error
     */
    pub fn add_instruction(
        ctx: Context<AddInstruction>,
        instruction_index: u8,
        program_id: Pubkey,
        accounts: Vec<ProposalAccountMeta>,
        data: Vec<u8>,
    ) -> Result<()> {
        instructions::proposal::add_instruction(
            ctx, instruction_index, program_id, accounts, data
        )
    }

    /**
     * Activates a proposal to begin the voting phase
     * 
     * @param ctx The context containing accounts for proposal activation
     * @return Result indicating success or specific error
     */
    pub fn activate_proposal(
        ctx: Context<ActivateProposal>,
    ) -> Result<()> {
        instructions::proposal::activate_proposal(ctx)
    }

    /**
     * Casts a vote on an active proposal
     * 
     * @param ctx The context containing accounts for voting
     * @param vote Vote choice
     * @param vote_weight Token weight to apply to the vote
     * @return Result indicating success or specific error
     */
    pub fn cast_vote(
        ctx: Context<CastVote>,
        vote: VoteKind,
        vote_weight: u64,
    ) -> Result<()> {
        instructions::voting::cast_vote(ctx, vote, vote_weight)
    }

    /**
     * Changes a previously cast vote if vote changes are allowed
     * 
     * @param ctx The context containing accounts for vote changing
     * @param vote New vote choice
     * @return Result indicating success or specific error
     */
    pub fn change_vote(
        ctx: Context<ChangeVote>,
        vote: VoteKind,
    ) -> Result<()> {
        instructions::voting::change_vote(ctx, vote)
    }

    /**
     * Relinquishes votes on a proposal before voting ends
     * 
     * @param ctx The context containing accounts for vote relinquishing
     * @return Result indicating success or specific error
     */
    pub fn relinquish_vote(
        ctx: Context<RelinquishVote>,
    ) -> Result<()> {
        instructions::voting::relinquish_vote(ctx)
    }

    /**
     * Finalizes a proposal after voting ends
     * 
     * @param ctx The context containing accounts for proposal finalization
     * @return Result indicating success or specific error
     */
    pub fn finalize_proposal(
        ctx: Context<FinalizeProposal>,
    ) -> Result<()> {
        instructions::proposal::finalize_proposal(ctx)
    }

    /**
     * Executes an instruction from an approved proposal
     * 
     * @param ctx The context containing accounts for instruction execution
     * @param instruction_index Index of the instruction to execute
     * @return Result indicating success or specific error
     */
    pub fn execute_instruction(
        ctx: Context<ExecuteInstruction>,
        instruction_index: u8,
    ) -> Result<()> {
        instructions::execution::execute_instruction(ctx, instruction_index)
    }

    /**
     * Releases a proposal from timelock, making it ready for execution
     * 
     * @param ctx The context containing accounts for timelock release
     * @return Result indicating success or specific error
     */
    pub fn release_timelock(
        ctx: Context<ReleaseTimelock>,
    ) -> Result<()> {
        instructions::execution::release_timelock(ctx)
    }

    /**
     * Delegates voting power to another account
     * 
     * @param ctx The context containing accounts for voting delegation
     * @param amount Amount of voting power to delegate
     * @return Result indicating success or specific error
     */
    pub fn delegate_voting_power(
        ctx: Context<DelegateVotingPower>,
        amount: u64,
    ) -> Result<()> {
        instructions::voting::delegate_voting_power(ctx, amount)
    }

    /**
     * Revokes a previous delegation of voting power
     * 
     * @param ctx The context containing accounts for delegation revocation
     * @param amount Amount of voting power to revoke
     * @return Result indicating success or specific error
     */
    pub fn revoke_voting_delegation(
        ctx: Context<RevokeVotingDelegation>,
        amount: u64,
    ) -> Result<()> {
        instructions::voting::revoke_voting_delegation(ctx, amount)
    }

    /**
     * Flags a proposal for further review or special handling
     * 
     * @param ctx The context containing accounts for flagging a proposal
     * @param flag_type Type of flag to apply
     * @param reason Reason for the flag
     * @return Result indicating success or specific error
     */
    pub fn flag_proposal(
        ctx: Context<FlagProposal>,
        flag_type: ProposalFlagType,
        reason: String,
    ) -> Result<()> {
        instructions::proposal::flag_proposal(ctx, flag_type, reason)
    }

    /**
     * Creates a required signatory record for a proposal
     * 
     * @param ctx The context containing accounts for adding a signatory
     * @param signatory Public key of the required signatory
     * @return Result indicating success or specific error
     */
    pub fn add_signatory(
        ctx: Context<AddSignatory>,
        signatory: Pubkey,
    ) -> Result<()> {
        instructions::proposal::add_signatory(ctx, signatory)
    }

    /**
     * Signs off on a proposal as a required signatory
     * 
     * @param ctx The context containing accounts for signing off
     * @return Result indicating success or specific error
     */
    pub fn sign_off_proposal(
        ctx: Context<SignOffProposal>,
    ) -> Result<()> {
        instructions::proposal::sign_off_proposal(ctx)
    }

    /**
     * Creates a comment on a proposal for on-chain discussion
     * 
     * @param ctx The context containing accounts for commenting
     * @param comment Comment text
     * @param reply_to Optional comment ID this is replying to
     * @return Result indicating success or specific error
     */
    pub fn post_comment(
        ctx: Context<PostComment>,
        comment: String,
        reply_to: Option<u64>,
    ) -> Result<()> {
        instructions::proposal::post_comment(ctx, comment, reply_to)
    }

    /**
     * Sets up a governance token distribution to multiple recipients
     * 
     * @param ctx The context containing accounts for token distribution
     * @param recipients List of recipient addresses and amounts
     * @param distribution_type Type of distribution (airdrop, grant, etc.)
     * @return Result indicating success or specific error
     */
    pub fn create_token_distribution(
        ctx: Context<CreateTokenDistribution>,
        recipients: Vec<TokenRecipient>,
        distribution_type: DistributionType,
    ) -> Result<()> {
        instructions::token::create_token_distribution(ctx, recipients, distribution_type)
    }

    /**
     * Claims tokens from a token distribution
     * 
     * @param ctx The context containing accounts for claiming tokens
     * @return Result indicating success or specific error
     */
    pub fn claim_distribution(
        ctx: Context<ClaimDistribution>,
    ) -> Result<()> {
        instructions::token::claim_distribution(ctx)
    }

    /**
     * Creates a time-weighted voting position with voting power that increases over time
     * 
     * @param ctx The context containing accounts for time lock creation
     * @param amount Amount of tokens to lock
     * @param lock_duration_seconds Duration to lock tokens for
     * @return Result indicating success or specific error
     */
    pub fn create_time_weighted_position(
        ctx: Context<CreateTimeWeightedPosition>,
        amount: u64,
        lock_duration_seconds: i64,
    ) -> Result<()> {
        instructions::voting::create_time_weighted_position(ctx, amount, lock_duration_seconds)
    }

    /**
     * Sets up governance authority over a program
     * 
     * @param ctx The context containing accounts for setting up governance
     * @param governance_type Type of governance authority
     * @return Result indicating success or specific error
     */
    pub fn set_governance_authority(
        ctx: Context<SetGovernanceAuthority>,
        governance_type: GovernanceType,
    ) -> Result<()> {
        instructions::execution::set_governance_authority(ctx, governance_type)
    }

    /**
     * Creates a snapshot of token holders for a historical voting record
     * 
     * @param ctx The context containing accounts for creating snapshot
     * @param description Description of what this snapshot is for
     * @return Result indicating success or specific error
     */
    pub fn create_voting_snapshot(
        ctx: Context<CreateVotingSnapshot>,
        description: String,
    ) -> Result<()> {
        instructions::voting::create_voting_snapshot(ctx, description)
    }

    /**
     * Sets the execution complexity level for a proposal
     * 
     * @param ctx The context containing accounts for setting complexity
     * @param complexity_level The complexity level to set
     * @param justification Justification for the complexity level
     * @return Result indicating success or specific error
     */
    pub fn set_proposal_complexity(
        ctx: Context<SetProposalComplexity>,
        complexity_level: ProposalComplexity,
        justification: String,
    ) -> Result<()> {
        instructions::proposal::set_proposal_complexity(ctx, complexity_level, justification)
    }

    /**
     * Vetos a proposal with appropriate authority
     * 
     * @param ctx The context containing accounts for vetoing
     * @param veto_reason Reason for the veto
     * @return Result indicating success or specific error
     */
    pub fn veto_proposal(
        ctx: Context<VetoProposal>,
        veto_reason: String,
    ) -> Result<()> {
        instructions::proposal::veto_proposal(ctx, veto_reason)
    }

    /**
     * Creates a special governance authority for emergency situations
     * 
     * @param ctx The context containing accounts for creating emergency authority
     * @param timeout_seconds How long this authority remains valid
     * @param allowed_actions Which actions this authority can perform
     * @return Result indicating success or specific error
     */
    pub fn create_emergency_authority(
        ctx: Context<CreateEmergencyAuthority>,
        timeout_seconds: i64,
        allowed_actions: EmergencyActions,
    ) -> Result<()> {
        instructions::security::create_emergency_authority(ctx, timeout_seconds, allowed_actions)
    }

    /**
     * Executes an emergency action with appropriate authority
     * 
     * @param ctx The context containing accounts for emergency execution
     * @param action The emergency action to perform
     * @param reason Justification for the emergency action
     * @return Result indicating success or specific error
     */
    pub fn execute_emergency_action(
        ctx: Context<ExecuteEmergencyAction>,
        action: EmergencyActionType,
        reason: String,
    ) -> Result<()> {
        instructions::security::execute_emergency_action(ctx, action, reason)
    }
}

// Context structs for program instructions
// Only stub definitions are provided here - full implementations would be in the instructions module

#[derive(Accounts)]
#[instruction(name: String, config: RealmConfig)]
pub struct CreateRealm<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    
    #[account(
        init,
        payer = payer,
        space = Realm::get_space(),
        seeds = [b"realm", name.as_bytes(), community_token_mint.key().as_ref()],
        bump
    )]
    pub realm: Account<'info, Realm>,
    
    pub community_token_mint: Account<'info, Mint>,
    
    pub council_token_mint: Option<Account<'info, Mint>>,
    
    pub system_program: Program<'info, System>,
    
    pub token_program: Program<'info, Token>,
    
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct UpdateRealmConfig<'info> {
    pub realm_authority: Signer<'info>,
    
    #[account(
        mut,
        has_one = authority @ GovernanceError::InvalidAuthority,
        seeds = [b"realm", &realm.name, realm.community_token_mint.as_ref()],
        bump = realm.bump
    )]
    pub realm: Account<'info, Realm>,
    
    pub authority: Pubkey,
}

// Additional context structs follow similar patterns...

// The file would continue with stub definitions for all instruction contexts
// Actual implementation details would be in the respective module files