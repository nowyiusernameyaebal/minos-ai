//! Proposal definitions for the governance program
//!
//! This module defines the proposal-related structures for the governance program,
//! including proposal types, actions, voting options, and weight calculations.
//! These structures provide a framework for creating and managing governance proposals.

use anchor_lang::prelude::*;
use std::collections::HashMap;

/// Represents the different types of governance proposals
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, Debug)]
pub enum ProposalType {
    /// Protocol parameter changes
    ParameterChange,
    
    /// Smart contract upgrades
    ProgramUpgrade,
    
    /// Treasury fund allocations
    TreasuryManagement,
    
    /// Integration with new protocols or services
    ProtocolIntegration,
    
    /// Emergency actions (e.g., pause functionality)
    Emergency,
    
    /// Changes to governance system itself
    GovernanceChange,
    
    /// General text proposals (non-executable)
    Text,
    
    /// Custom action type
    Custom(String),
}

/// Represents the current state of a proposal
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, Debug)]
pub enum ProposalState {
    /// Proposal is being drafted and can be modified
    Draft,
    
    /// Proposal is active and accepting votes
    Active,
    
    /// Proposal succeeded (met quorum and approval threshold)
    Succeeded,
    
    /// Proposal was defeated (failed to meet quorum or approval threshold)
    Defeated,
    
    /// Proposal was successfully executed
    Executed,
    
    /// Proposal was cancelled before completion
    Cancelled,
    
    /// Proposal was vetoed by a special authority
    Vetoed,
    
    /// Proposal expired without being executed
    Expired,
}

/// Represents voting options
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, Debug)]
pub enum VoteOption {
    /// Vote in favor of the proposal
    Yes,
    
    /// Vote against the proposal
    No,
    
    /// Abstain from voting (counts for quorum but not approval)
    Abstain,
}

/// Represents vote weight for weighted voting
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct VoteWeight {
    /// The raw token amount used for voting
    pub raw_amount: u64,
    
    /// The adjusted voting weight after applying multipliers
    pub adjusted_weight: u64,
    
    /// Any delegation info for this vote
    pub delegated_from: Option<Pubkey>,
}

/// Represents executable actions for proposals
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum ProposalAction {
    /// Change a protocol parameter
    ChangeParameter {
        /// The program to target
        program_id: Pubkey,
        
        /// The parameter name
        parameter_name: String,
        
        /// The new value as serialized bytes
        new_value: Vec<u8>,
    },
    
    /// Upgrade a program
    UpgradeProgram {
        /// The program to upgrade
        program_id: Pubkey,
        
        /// The buffer containing the new program
        buffer_address: Pubkey,
        
        /// The spill address for rent
        spill_address: Pubkey,
    },
    
    /// Transfer treasury funds
    TransferFunds {
        /// The source token account
        source: Pubkey,
        
        /// The destination token account
        destination: Pubkey,
        
        /// The amount to transfer
        amount: u64,
    },
    
    /// Add a new strategy to the protocol
    AddStrategy {
        /// The strategy name
        name: String,
        
        /// The strategy implementation program
        program_id: Pubkey,
        
        /// Strategy parameters
        parameters: HashMap<String, Vec<u8>>,
    },
    
    /// Pause a specific protocol function
    PauseFunction {
        /// The program to target
        program_id: Pubkey,
        
        /// The function to pause
        function_name: String,
    },
    
    /// Unpause a specific protocol function
    UnpauseFunction {
        /// The program to target
        program_id: Pubkey,
        
        /// The function to unpause
        function_name: String,
    },
    
    /// Change a governance parameter
    ChangeGovernance {
        /// The parameter name
        parameter_name: String,
        
        /// The new value as serialized bytes
        new_value: Vec<u8>,
    },
    
    /// Custom action with arbitrary instructions
    CustomInstruction {
        /// Program ID to call
        program_id: Pubkey,
        
        /// Account metas for the instruction
        accounts: Vec<AccountMetaData>,
        
        /// Instruction data
        data: Vec<u8>,
    },
}

/// Account metadata for custom instructions
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct AccountMetaData {
    /// Public key of the account
    pub pubkey: Pubkey,
    
    /// Whether the account is a signer
    pub is_signer: bool,
    
    /// Whether the account is writable
    pub is_writable: bool,
}

/// Complete proposal data structure
#[account]
pub struct Proposal {
    /// The address of the proposer
    pub proposer: Pubkey,
    
    /// Unique proposal index
    pub index: u64,
    
    /// Type of proposal
    pub proposal_type: ProposalType,
    
    /// Current state of the proposal
    pub state: ProposalState,
    
    /// Proposal title (limited to MAX_TITLE_LENGTH)
    pub title: String,
    
    /// Proposal description (limited to MAX_DESCRIPTION_LENGTH)
    pub description: String,
    
    /// Timestamp when the proposal was created
    pub created_at: i64,
    
    /// Timestamp when voting starts
    pub voting_starts_at: i64,
    
    /// Timestamp when voting ends
    pub voting_ends_at: i64,
    
    /// Timestamp when the proposal can be executed (after timelock)
    pub execution_timestamp: i64,
    
    /// Number of YES votes
    pub yes_votes: u64,
    
    /// Number of NO votes
    pub no_votes: u64,
    
    /// Number of ABSTAIN votes
    pub abstain_votes: u64,
    
    /// Total voting power used on this proposal
    pub total_voting_power: u64,
    
    /// Voting period duration in seconds
    pub voting_period: i64,
    
    /// Execution delay (timelock) in seconds
    pub execution_delay: i64,
    
    /// List of executable actions
    pub actions: Vec<ProposalAction>,
}

impl Proposal {
    /// Calculate voting results
    pub fn calculate_results(&self) -> (bool, bool) {
        // Total votes that count towards quorum
        let total_votes_for_quorum = self.yes_votes
            .saturating_add(self.no_votes)
            .saturating_add(self.abstain_votes);
        
        // Total votes that count towards approval
        let total_votes_for_approval = self.yes_votes.saturating_add(self.no_votes);
        
        // Calculate approval percentage
        let approval_percentage = if total_votes_for_approval > 0 {
            (self.yes_votes as f64 / total_votes_for_approval as f64) * 100.0
        } else {
            0.0
        };
        
        // Determine if quorum and approval threshold are met
        // Note: This requires the governance configuration to determine thresholds
        let quorum_met = total_votes_for_quorum > 0; // Placeholder - implement actual logic
        let approval_met = approval_percentage >= 50.0; // Placeholder - implement actual logic
        
        (quorum_met, approval_met)
    }
    
    /// Get the estimated proposal execution time
    pub fn get_execution_time(&self) -> Option<i64> {
        match self.state {
            ProposalState::Succeeded | ProposalState::Executed => Some(self.execution_timestamp),
            _ => None,
        }
    }
    
    /// Check if the proposal is in an active voting state
    pub fn is_voting_active(&self, current_time: i64) -> bool {
        self.state == ProposalState::Active && 
        current_time >= self.voting_starts_at && 
        current_time <= self.voting_ends_at
    }
    
    /// Check if the proposal can be executed
    pub fn is_executable(&self, current_time: i64) -> bool {
        self.state == ProposalState::Succeeded && 
        current_time >= self.execution_timestamp
    }
    
    /// Check if the proposal has expired
    pub fn is_expired(&self, current_time: i64, expiration_period: i64) -> bool {
        self.state == ProposalState::Succeeded && 
        current_time > self.execution_timestamp.saturating_add(expiration_period)
    }
    
    /// Calculate the vote participation percentage
    pub fn get_participation_percentage(&self, total_supply: u64) -> f64 {
        if total_supply == 0 {
            return 0.0;
        }
        
        let total_votes = self.yes_votes
            .saturating_add(self.no_votes)
            .saturating_add(self.abstain_votes);
            
        (total_votes as f64 / total_supply as f64) * 100.0
    }
}

/// ProposalActionBatch allows multiple actions to be executed atomically
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct ProposalActionBatch {
    /// List of actions to execute together
    pub actions: Vec<ProposalAction>,
    
    /// Whether all actions must succeed or the batch fails
    pub require_all_success: bool,
}

/// Custom implementation helpers for ProposalAction
impl ProposalAction {
    /// Get a human-readable description of the action
    pub fn get_description(&self) -> String {
        match self {
            ProposalAction::ChangeParameter { parameter_name, .. } => {
                format!("Change parameter: {}", parameter_name)
            }
            ProposalAction::UpgradeProgram { program_id, .. } => {
                format!("Upgrade program: {}", program_id)
            }
            ProposalAction::TransferFunds { amount, .. } => {
                format!("Transfer {} tokens", amount)
            }
            ProposalAction::AddStrategy { name, .. } => {
                format!("Add strategy: {}", name)
            }
            ProposalAction::PauseFunction { function_name, .. } => {
                format!("Pause function: {}", function_name)
            }
            ProposalAction::UnpauseFunction { function_name, .. } => {
                format!("Unpause function: {}", function_name)
            }
            ProposalAction::ChangeGovernance { parameter_name, .. } => {
                format!("Change governance parameter: {}", parameter_name)
            }
            ProposalAction::CustomInstruction { .. } => {
                "Execute custom instruction".to_string()
            }
        }
    }
    
    /// Check if this action requires a higher approval threshold
    pub fn requires_higher_threshold(&self) -> bool {
        matches!(
            self,
            ProposalAction::UpgradeProgram { .. } |
            ProposalAction::ChangeGovernance { .. } |
            ProposalAction::TransferFunds { amount, .. } if *amount > 10_000_000_000
        )
    }
    
    /// Check if this action requires a longer timelock
    pub fn requires_longer_timelock(&self) -> bool {
        matches!(
            self,
            ProposalAction::UpgradeProgram { .. } |
            ProposalAction::ChangeGovernance { .. }
        )
    }
}