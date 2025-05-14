-> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"proposal",
                realm.as_ref(),
                governing_token_mint.as_ref(),
                &proposal_index.to_le_bytes(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Realm
        32 +
        // Governing token mint
        32 +
        // State
        1 +
        // Token owner record
        32 +
        // Name/title
        MAX_PROPOSAL_TITLE_LENGTH +
        // Description
        MAX_PROPOSAL_DESCRIPTION_LENGTH +
        // Voting start time
        8 +
        // Voting end time
        8 +
        // Execution start time option
        1 + 8 +
        // Execution expiry time option
        1 + 8 +
        // Vote threshold percentage
        1 +
        // Voting results (approximate serialized size)
        100 +
        // Instruction count
        1 +
        // Executed instruction count
        1 +
        // Required signatory count
        1 +
        // Signed off count
        1 +
        // Draft version
        1 +
        // Flag
        1 +
        // External link option (approximate)
        1 + 100 +
        // Voting type (approximate)
        10 +
        // Execution type
        1 +
        // Complexity
        1 +
        // Created at timestamp
        8 +
        // Updated at timestamp
        8 +
        // Version
        1 +
        // Options (approximate)
        1 + 200 +
        // Reserved space
        32 +
        // Bump seed
        1
    }
}

impl ProposalInstruction {
    /// Calculate instruction PDA
    pub fn find_address(
        program_id: &Pubkey,
        proposal: &Pubkey,
        instruction_index: u8,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"proposal-instruction",
                proposal.as_ref(),
                &[instruction_index],
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account based on instruction data size
    pub fn get_space(data_size: usize, account_metas_count: usize) -> usize {
        // Discriminator
        8 +
        // Proposal pubkey
        32 +
        // Instruction index
        1 +
        // Program ID
        32 +
        // Accounts vec prefix and elements
        4 + (account_metas_count * size_of::<ProposalAccountMeta>()) +
        // Data vec prefix and elements
        4 + data_size +
        // Executed flag
        1 +
        // Execution time option
        1 + 8 +
        // Executor option
        1 + 32 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
}

impl VoteRecord {
    /// Calculate vote record PDA
    pub fn find_address(
        program_id: &Pubkey,
        proposal: &Pubkey,
        token_owner_record: &Pubkey,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"vote-record",
                proposal.as_ref(),
                token_owner_record.as_ref(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Proposal pubkey
        32 +
        // Token owner record
        32 +
        // Vote (with maximum enum size)
        10 +
        // Vote weight
        8 +
        // Is relinquished flag
        1 +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
}

impl TokenOwnerRecord {
    /// Calculate token owner record PDA
    pub fn find_address(
        program_id: &Pubkey,
        realm: &Pubkey,
        governing_token_mint: &Pubkey,
        governing_token_owner: &Pubkey,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"token-owner-record",
                realm.as_ref(),
                governing_token_mint.as_ref(),
                governing_token_owner.as_ref(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Realm
        32 +
        // Governing token mint
        32 +
        // Governing token owner
        32 +
        // Governing token deposit account
        32 +
        // Governing token deposit amount
        8 +
        // Unrelinquished votes count
        4 +
        // Is council token flag
        1 +
        // Governance delegate option
        1 + 32 +
        // Delegated voting power
        8 +
        // Received delegated voting power
        8 +
        // Created at timestamp
        8 +
        // Updated at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        32 +
        // Bump seed
        1
    }
}

impl SignatoryRecord {
    /// Calculate signatory record PDA
    pub fn find_address(
        program_id: &Pubkey,
        proposal: &Pubkey,
        signatory: &Pubkey,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"signatory-record",
                proposal.as_ref(),
                signatory.as_ref(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Proposal
        32 +
        // Signatory
        32 +
        // Signed off flag
        1 +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
}

impl CommentRecord {
    /// Calculate comment record PDA
    pub fn find_address(
        program_id: &Pubkey,
        proposal: &Pubkey,
        comment_id: u64,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"comment-record",
                proposal.as_ref(),
                &comment_id.to_le_bytes(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Proposal
        32 +
        // Author
        32 +
        // Comment text
        MAX_COMMENT_LENGTH +
        // Reply to option
        1 + 8 +
        // Comment ID
        8 +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
    
    /// Get the comment text as a string
    pub fn get_comment_text(&self) -> String {
        let mut comment_bytes = self.comment.to_vec();
        let null_index = comment_bytes.iter().position(|&b| b == 0).unwrap_or(comment_bytes.len());
        comment_bytes.truncate(null_index);
        String::from_utf8(comment_bytes).unwrap_or_else(|_| "Invalid UTF-8".to_string())
    }
}

impl EmergencyAuthorityRecord {
    /// Calculate emergency authority record PDA
    pub fn find_address(
        program_id: &Pubkey,
        realm: &Pubkey,
        authority: &Pubkey,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"emergency-authority",
                realm.as_ref(),
                authority.as_ref(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Realm
        32 +
        // Authority
        32 +
        // Allowed actions
        4 +
        // Expiry time
        8 +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
    
    /// Check if the emergency authority is still valid
    pub fn is_valid(&self) -> bool {
        let clock = Clock::get().unwrap();
        let current_time = clock.unix_timestamp;
        
        current_time < self.expiry_time
    }
}

impl TokenDistribution {
    /// Calculate token distribution PDA
    pub fn find_address(
        program_id: &Pubkey,
        realm: &Pubkey,
        token_mint: &Pubkey,
        distribution_index: u64,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"token-distribution",
                realm.as_ref(),
                token_mint.as_ref(),
                &distribution_index.to_le_bytes(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account with given number of recipients
    pub fn get_space(recipient_count: usize) -> usize {
        // Discriminator
        8 +
        // Realm
        32 +
        // Token mint
        32 +
        // Authority
        32 +
        // Distribution type
        1 +
        // Recipients vec prefix
        4 +
        // Recipients (size of each recipient * count)
        (recipient_count * (32 + 8 + 9 + 9)) +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Reserved space
        8 +
        // Bump seed
        1
    }
}

/// Constants module - system-wide governance constants
pub mod constants {
    /// Maximum number of concurrent proposals per realm
    pub const MAX_CONCURRENT_PROPOSALS: u16 = 100;
    
    /// Maximum voting delegation depth
    pub const MAX_DELEGATION_DEPTH: u8 = 1;
    
    /// Minimum weight for creating proposals (1%)
    pub const MIN_PROPOSAL_CREATION_WEIGHT: u16 = 100;
    
    /// Minimum quorum percentage
    pub const MIN_QUORUM_PERCENTAGE: u8 = 1;
    
    /// Minimum vote threshold percentage
    pub const MIN_VOTE_THRESHOLD_PERCENTAGE: u8 = 1;
    
    /// Maximum proposal name length
    pub const MAX_PROPOSAL_NAME_LENGTH: usize = 100;
    
    /// Seed prefix for governance PDAs
    pub const GOVERNANCE_SEED_PREFIX: &[u8] = b"governance";
}

/// Utility functions for governance operations
pub mod utils {
    use super::*;
    
    /// Calculate the time-weighted voting power based on configuration and lock duration
    pub fn calculate_time_weighted_power(
        token_amount: u64, 
        lock_duration_seconds: i64,
        max_multiplier_bps: u16,
        seconds_for_max_boost: u64,
    ) -> u64 {
        // Early return for zero amounts
        if token_amount == 0 {
            return 0;
        }
        
        // Calculate the boost based on lock duration
        let max_multiplier_factor = max_multiplier_bps as f64 / BASIS_POINTS_DIVISOR as f64;
        
        // Cap the lock duration at the maximum
        let effective_lock_duration = min(lock_duration_seconds as u64, seconds_for_max_boost);
        
        // Calculate the boost factor (linear between 1.0 and max_multiplier_factor)
        let boost_factor = 1.0 + (max_multiplier_factor - 1.0) * 
                           (effective_lock_duration as f64 / seconds_for_max_boost as f64);
        
        // Apply the boost to the token amount
        let weighted_power = (token_amount as f64 * boost_factor) as u64;
        
        weighted_power
    }
    
    /// Calculate vote results based on vote counts and configuration
    pub fn calculate_vote_results(
        yes_votes: u64,
        no_votes: u64,
        abstain_votes: u64,
        veto_votes: u64,
        total_supply: u64,
        vote_threshold_percentage: u8,
        quorum_percentage: u8,
        threshold_type: VoteThresholdType,
    ) -> VotingResults {
        let mut results = VotingResults::default();
        
        // Set the vote counts
        results.yes_votes_count = yes_votes;
        results.no_votes_count = no_votes;
        results.abstain_votes_count = abstain_votes;
        results.veto_votes_count = veto_votes;
        
        // Calculate total votes cast
        let total_votes_cast = yes_votes + no_votes + abstain_votes;
        
        // Check if there's a veto
        if veto_votes > 0 {
            // A veto automatically fails the proposal
            results.threshold_achieved = false;
            return results;
        }
        
        // Check if quorum is achieved
        let min_quorum_votes = (total_supply as u128 * quorum_percentage as u128 / 100) as u64;
        results.quorum_achieved = total_votes_cast >= min_quorum_votes;
        
        // If quorum is not achieved, the proposal automatically fails
        if !results.quorum_achieved {
            results.threshold_achieved = false;
            return results;
        }
        
        // Calculate if the threshold is achieved based on the threshold type
        let threshold_denominator = match threshold_type {
            VoteThresholdType::YesVotePercentage => yes_votes + no_votes,
            VoteThresholdType::SupplyPercentage => total_supply,
            VoteThresholdType::VotePercentage => total_votes_cast,
        };
        
        // Avoid division by zero
        if threshold_denominator == 0 {
            results.threshold_achieved = false;
            return results;
        }
        
        // Calculate the percentage of yes votes
        let yes_vote_percentage = (yes_votes as u128 * 100 as u128 / threshold_denominator as u128) as u8;
        
        // Check if the threshold is achieved
        results.threshold_achieved = yes_vote_percentage >= vote_threshold_percentage;
        
        results
    }
    
    /// Calculate quadratic voting power
    pub fn calculate_quadratic_power(token_amount: u64) -> u64 {
        // Quadratic voting power is the square root of the token amount
        // Using u64 precision by scaling up before sqrt and down after
        let scaled_amount = (token_amount as f64).sqrt() as u64;
        scaled_amount
    }
    
    /// Sanitize a proposal name to ensure it meets requirements
    pub fn sanitize_proposal_name(name: &str) -> Result<[u8; MAX_PROPOSAL_TITLE_LENGTH], GovernanceError> {
        if name.is_empty() {
            return Err(GovernanceError::InvalidProposalName);
        }
        
        let bytes = name.as_bytes();
        if bytes.len() > MAX_PROPOSAL_TITLE_LENGTH {
            return Err(GovernanceError::ProposalNameTooLong);
        }
        
        let mut name_bytes = [0u8; MAX_PROPOSAL_TITLE_LENGTH];
        name_bytes[..bytes.len()].copy_from_slice(bytes);
        
        Ok(name_bytes)
    }
    
    /// Sanitize a proposal description to ensure it meets requirements
    pub fn sanitize_proposal_description(description: &str) -> Result<[u8; MAX_PROPOSAL_DESCRIPTION_LENGTH], GovernanceError> {
        if description.is_empty() {
            return Err(GovernanceError::InvalidProposalDescription);
        }
        
        let bytes = description.as_bytes();
        if bytes.len() > MAX_PROPOSAL_DESCRIPTION_LENGTH {
            return Err(GovernanceError::ProposalDescriptionTooLong);
        }
        
        let mut description_bytes = [0u8; MAX_PROPOSAL_DESCRIPTION_LENGTH];
        description_bytes[..bytes.len()].copy_from_slice(bytes);
        
        Ok(description_bytes)
    }
}
/**
 * @title Governance State Data Structures
 * @notice Defines the on-chain state accounts and data structures for the Minos-AI governance protocol
 * @dev This module contains all data structures representing the persistent state of the governance
 *      program, including realms, proposals, votes, token deposits, and execution state
 * 
 * @custom:security-contact security@minos-ai.com
 * @author Minos-AI Governance Team
 * @date January 12, 2025
 */

use anchor_lang::prelude::*;
use anchor_spl::token::{TokenAccount, Mint};
use borsh::{BorshDeserialize, BorshSerialize};
use std::{
    cmp::{max, min},
    convert::TryFrom,
    mem::size_of,
};
use crate::{constants::*, errors::GovernanceError};

/// Maximum length for realm and governance names
pub const MAX_NAME_LENGTH: usize = 32;

/// Maximum length for proposal titles
pub const MAX_PROPOSAL_TITLE_LENGTH: usize = 128;

/// Maximum length for proposal descriptions
pub const MAX_PROPOSAL_DESCRIPTION_LENGTH: usize = 4096;

/// Maximum length for comment text
pub const MAX_COMMENT_LENGTH: usize = 1024;

/// Maximum number of instructions allowed in a single proposal
pub const MAX_INSTRUCTION_COUNT: u8 = 24;

/// Maximum number of signatories allowed for a proposal
pub const MAX_SIGNATORIES: u8 = 10;

/// Maximum number of options in a multi-choice vote
pub const MAX_VOTE_OPTIONS: u8 = 10;

/// Minimum timelock period in seconds (24 hours)
pub const MIN_TIMELOCK_SECONDS: i64 = 86400;

/// Maximum timelock period in seconds (30 days)
pub const MAX_TIMELOCK_SECONDS: i64 = 2592000;

/// Minimum voting period in seconds (3 days)
pub const MIN_VOTING_PERIOD_SECONDS: i64 = 259200;

/// Maximum voting period in seconds (14 days)
pub const MAX_VOTING_PERIOD_SECONDS: i64 = 1209600;

/// Minimum value for vote threshold percentage (1%)
pub const MIN_VOTE_THRESHOLD_PERCENTAGE: u8 = 1;

/// Maximum value for vote threshold percentage (100%)
pub const MAX_VOTE_THRESHOLD_PERCENTAGE: u8 = 100;

/// Default quorum percentage (20%)
pub const DEFAULT_QUORUM_PERCENTAGE: u8 = 20;

/// Default voting period in seconds (5 days)
pub const DEFAULT_VOTING_PERIOD_SECONDS: i64 = 432000;

/// Default timelock period in seconds (2 days)
pub const DEFAULT_TIMELOCK_SECONDS: i64 = 172800;

/// Emergency action timeout (24 hours)
pub const EMERGENCY_ACTION_TIMEOUT_SECONDS: i64 = 86400;

/// Version identifier for current data structures 
pub const CURRENT_STATE_VERSION: u8 = 1;

/// Basis points for percentage calculations (100% = 10000)
pub const BASIS_POINTS_DIVISOR: u16 = 10000;

/**
 * VoteKind enum represents the possible vote choices for governance proposals
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum VoteKind {
    /// Vote in favor of the proposal
    Approve,
    
    /// Vote against the proposal
    Reject,
    
    /// Abstain from voting (counts for quorum but not approval)
    Abstain,
    
    /// Veto the proposal (governance admin only)
    Veto,
    
    /// Vote for a specific option in a multi-choice proposal
    MultiChoice {
        /// Index of the option being voted for
        option_index: u8,
    },
}

/**
 * VotingType enum represents different voting systems that can be used for proposals
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum VotingType {
    /// Standard single-choice yes/no voting
    SingleChoice,
    
    /// Multiple choice voting with several options
    MultiChoice {
        /// Number of options available in this vote
        option_count: u8,
        /// Whether "None of the above" is included as an option
        include_none_option: bool,
    },
    
    /// Ranked choice voting with preference ordering
    RankedChoice {
        /// Number of options available for ranking
        option_count: u8,
    },
    
    /// Quadratic voting where cost increases quadratically with votes
    Quadratic {
        /// Maximum voting power per voter
        max_voter_options: u8,
    },
    
    /// Approval voting where voters can approve multiple options
    ApprovalVoting {
        /// Number of options available
        option_count: u8, 
        /// Maximum number of approvals per voter
        max_approvals_per_voter: u8,
    },
}

/**
 * VoteWeightingType enum represents different ways to calculate voting weight
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum VoteWeightingType {
    /// One token equals one vote
    TokenCount,
    
    /// Square root of token count (reduces whale influence)
    Quadratic,
    
    /// Token count multiplied by lock duration
    TimeWeighted {
        /// Maximum time multiplier in basis points (e.g. 30000 = 3x max multiplier)
        max_time_multiplier_bps: u16,
        /// Time (in seconds) required to reach maximum multiplier
        seconds_for_max_boost: u64,
    },
    
    /// Custom weighting function
    Custom {
        /// Identifier for the custom weighting function
        weighting_function_id: u8,
    },
}

/**
 * ProposalStatus enum tracks the lifecycle state of a governance proposal
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum ProposalStatus {
    /// Initial state when created
    Draft,
    
    /// Cancelled before voting began
    Cancelled,
    
    /// Awaiting signoff from required signatories
    SigningOff,
    
    /// Open for voting
    Voting,
    
    /// Voting has ended, awaiting finalization
    VotingEnded,
    
    /// Proposal was defeated in voting
    Defeated,
    
    /// Proposal was approved but is in timelock
    Approved,
    
    /// Proposal has passed timelock and is ready for execution
    ExecutionReady,
    
    /// Proposal has been partially executed
    Executing,
    
    /// Proposal has been fully executed
    Completed,
    
    /// Proposal was vetoed by governance authority
    Vetoed,
    
    /// Execution window expired without completion
    Expired,
}

/**
 * ProposalFlagType enum identifies special review flags for proposals
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum ProposalFlagType {
    /// Normal proposal with no special flags
    None,
    
    /// Proposal contains high-risk instructions requiring special review
    HighRisk,
    
    /// Proposal may have a governance security impact
    SecurityImpact,
    
    /// Proposal contains invalid or potentially malicious instructions
    PotentiallyMalicious,
    
    /// Proposal is under community review for other reasons
    CommunityReview,
}

/**
 * ProposalComplexity enum categorizes proposals by execution complexity and risk
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum ProposalComplexity {
    /// Simple parameter change with low risk
    Low,
    
    /// Standard proposal with moderate complexity
    Medium,
    
    /// Complex multi-instruction proposal requiring careful review
    High,
    
    /// Critical protocol change with significant impact
    Critical,
}

/**
 * GovernanceTokenType enum identifies types of governance tokens
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum GovernanceTokenType {
    /// Standard fungible governance token
    Standard,
    
    /// Non-transferable membership token (1 per member)
    Membership,
    
    /// Token representing special council/admin rights
    Council,
    
    /// Token with vesting schedule for team allocation
    Vesting,
}

/**
 * GovernanceType enum identifies types of governance authority
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum GovernanceType {
    /// Standard governance with normal rules
    Standard,
    
    /// Emergency/multisig governance for critical issues
    Emergency,
    
    /// Special governance for protocol upgrades only
    Upgrade,
    
    /// Governance with delegated authority for specific domains
    Delegated,
}

/**
 * EmergencyActionType enum identifies types of emergency actions
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum EmergencyActionType {
    /// Pause specific protocol functionality
    Pause,
    
    /// Resume paused functionality
    Resume,
    
    /// Emergency withdrawal of funds
    Withdraw,
    
    /// Override a governance decision
    Override,
    
    /// Apply an emergency patch
    Patch,
}

/**
 * DistributionType enum identifies types of token distributions
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum DistributionType {
    /// Initial token distribution/airdrop
    Initial,
    
    /// Grants for contributors
    Grant,
    
    /// Rewards for participation
    Reward,
    
    /// Retroactive distribution
    Retroactive,
}

/**
 * TokenRecipient struct defines a recipient in a token distribution
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct TokenRecipient {
    /// Recipient address
    pub recipient: Pubkey,
    
    /// Amount to distribute
    pub amount: u64,
    
    /// Optional vesting duration in seconds
    pub vesting_seconds: Option<u64>,
    
    /// Optional cliff duration in seconds
    pub cliff_seconds: Option<u64>,
}

/**
 * EmergencyActions struct defines allowed emergency actions as bit flags
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default)]
pub struct EmergencyActions {
    /// Bit flags for allowed actions (1=Pause, 2=Resume, 4=Withdraw, etc.)
    pub allowed_actions: u32,
}

impl EmergencyActions {
    /// Creates a new EmergencyActions with all permissions
    pub fn all() -> Self {
        Self { allowed_actions: u32::MAX }
    }
    
    /// Creates a new EmergencyActions with no permissions
    pub fn none() -> Self {
        Self { allowed_actions: 0 }
    }
    
    /// Checks if a specific action is allowed
    pub fn is_allowed(&self, action: EmergencyActionType) -> bool {
        let flag = match action {
            EmergencyActionType::Pause => 1,
            EmergencyActionType::Resume => 2,
            EmergencyActionType::Withdraw => 4,
            EmergencyActionType::Override => 8,
            EmergencyActionType::Patch => 16,
        };
        
        (self.allowed_actions & flag) != 0
    }
}

/**
 * RealmConfig struct contains configuration parameters for a governance realm
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct RealmConfig {
    /// Minimum percentage of votes required for approval (1-100)
    pub vote_threshold_percentage: u8,
    
    /// Minimum percentage of supply that must vote for quorum (1-100)
    pub quorum_percentage: u8,
    
    /// Minimum delay in seconds after approval before execution
    pub timelock_seconds: i64,
    
    /// How long proposals remain open for voting (in seconds)
    pub voting_period_seconds: i64,
    
    /// Minimum amount of governance tokens needed to create proposals
    pub min_proposal_creation_threshold: u64,
    
    /// Whether token deposits are required to create proposals
    pub proposal_creation_requires_deposit: bool,
    
    /// Amount required for proposal deposits (if enabled)
    pub proposal_deposit_amount: u64,
    
    /// Type of vote counting used for this realm
    pub voting_type: VotingType,
    
    /// How voting weight is calculated
    pub vote_weighting_type: VoteWeightingType,
    
    /// How the vote threshold is calculated
    pub vote_threshold_type: VoteThresholdType,
    
    /// Whether vote changes are allowed after casting
    pub allow_vote_changes: bool,
    
    /// Maximum time a proposal can remain in draft state
    pub max_draft_time_seconds: i64,
    
    /// Whether governance tokens can be delegated
    pub delegation_enabled: bool,
    
    /// Whether vote tipping is enabled (early finalization when outcome certain)
    pub vote_tipping_enabled: bool,
    
    /// Whether multiple instructions per proposal are allowed
    pub multiple_instructions_enabled: bool,
    
    /// Whether early execution is allowed
    pub early_execution_enabled: bool,
    
    /// Whether council veto is enabled
    pub council_veto_enabled: bool,
    
    /// Maximum number of options for multi-choice votes
    pub max_option_count: u8,
    
    /// Version of the config format for upgrade compatibility
    pub version: u8,
}

impl Default for RealmConfig {
    fn default() -> Self {
        Self {
            vote_threshold_percentage: 50,
            quorum_percentage: DEFAULT_QUORUM_PERCENTAGE,
            timelock_seconds: DEFAULT_TIMELOCK_SECONDS,
            voting_period_seconds: DEFAULT_VOTING_PERIOD_SECONDS,
            min_proposal_creation_threshold: 100_000_000, // 100 tokens with 6 decimals
            proposal_creation_requires_deposit: false,
            proposal_deposit_amount: 0,
            voting_type: VotingType::SingleChoice,
            vote_weighting_type: VoteWeightingType::TokenCount,
            vote_threshold_type: VoteThresholdType::YesVotePercentage,
            allow_vote_changes: true,
            max_draft_time_seconds: 604800, // 7 days
            delegation_enabled: true,
            vote_tipping_enabled: false,
            multiple_instructions_enabled: true,
            early_execution_enabled: false,
            council_veto_enabled: false,
            max_option_count: 5,
            version: CURRENT_STATE_VERSION,
        }
    }
}

/**
 * VoteThresholdType enum defines how vote thresholds are calculated
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum VoteThresholdType {
    /// Yes votes as percentage of (Yes + No) votes
    YesVotePercentage,
    
    /// Yes votes as percentage of all possible votes
    SupplyPercentage,
    
    /// Yes votes as percentage of all cast votes including abstain
    VotePercentage,
}

/**
 * VotingResults struct tracks vote counts for a proposal
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default)]
pub struct VotingResults {
    /// Number of Yes votes
    pub yes_votes_count: u64,
    
    /// Number of No votes
    pub no_votes_count: u64,
    
    /// Number of Abstain votes
    pub abstain_votes_count: u64,
    
    /// Number of veto votes (admin only)
    pub veto_votes_count: u64,
    
    /// Number of unique voters who participated
    pub voter_count: u32,
    
    /// Whether quorum was reached
    pub quorum_achieved: bool,
    
    /// Whether the vote threshold was reached
    pub threshold_achieved: bool,
    
    /// Vote counts for each option in multi-choice proposals
    pub option_vote_counts: Option<Vec<u64>>,
    
    /// Index of the winning option in multi-choice proposals
    pub winning_option_index: Option<u8>,
}

/**
 * ProposalAccountMeta struct represents an account in a proposal instruction
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct ProposalAccountMeta {
    /// Public key of the account
    pub pubkey: Pubkey,
    
    /// Whether the account is a signer
    pub is_signer: bool,
    
    /// Whether the account is writable
    pub is_writable: bool,
}

/**
 * Realm account is the top-level governance container
 */
#[account]
#[derive(Debug)]
pub struct Realm {
    /// Account allowed to update realm config
    pub authority: Option<Pubkey>,
    
    /// Main governance token mint
    pub community_token_mint: Pubkey,
    
    /// Optional council token mint (for special voting rights)
    pub council_token_mint: Option<Pubkey>,
    
    /// Name of the realm (UTF-8 encoded)
    pub name: [u8; MAX_NAME_LENGTH],
    
    /// Configuration parameters
    pub config: RealmConfig,
    
    /// Number of proposals created in this realm
    pub proposal_count: u64,
    
    /// When the realm was created
    pub created_at: i64,
    
    /// Version of the Realm account structure
    pub version: u8,
    
    /// Whether the realm is currently active
    pub is_active: bool,
    
    /// Reserved for future extensions
    pub reserved: [u8; 64],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * Proposal account represents a governance proposal
 */
#[account]
#[derive(Debug)]
pub struct Proposal {
    /// Governance realm this proposal belongs to
    pub realm: Pubkey,
    
    /// Governance token mint used for voting
    pub governing_token_mint: Pubkey,
    
    /// Current state of the proposal
    pub state: ProposalStatus,
    
    /// Token owner record of the proposer
    pub token_owner_record: Pubkey,
    
    /// Name/title of the proposal
    pub name: [u8; MAX_PROPOSAL_TITLE_LENGTH],
    
    /// Detailed description of the proposal
    pub description: [u8; MAX_PROPOSAL_DESCRIPTION_LENGTH],
    
    /// When voting begins
    pub voting_start_time: i64,
    
    /// When voting ends
    pub voting_end_time: i64,
    
    /// When execution will be allowed after approval
    pub execution_start_time: Option<i64>,
    
    /// When execution window ends
    pub execution_expiry_time: Option<i64>,
    
    /// Vote threshold for this specific proposal
    pub vote_threshold_percentage: u8,
    
    /// Voting results
    pub voting_results: VotingResults,
    
    /// Number of proposal instructions
    pub instruction_count: u8,
    
    /// Number of executed instructions
    pub executed_instruction_count: u8,
    
    /// Number of required signatories
    pub required_signatory_count: u8,
    
    /// Number of signatories who have signed
    pub signed_off_count: u8,
    
    /// Draft proposal version number
    pub draft_version: u8,
    
    /// Flag indicating special handling needs
    pub flag: ProposalFlagType,
    
    /// Link to external resources (IPFS hash, URL, etc.)
    pub external_link: Option<String>,
    
    /// Type of voting for this proposal
    pub voting_type: VotingType,
    
    /// How execution is handled
    pub execution_type: ExecutionType,
    
    /// Complexity level for this proposal
    pub complexity: ProposalComplexity,
    
    /// When the proposal was created
    pub created_at: i64,
    
    /// When the proposal was last updated
    pub updated_at: i64,
    
    /// Version of the Proposal account structure
    pub version: u8,
    
    /// Multi-choice vote options
    pub options: Option<Vec<String>>,
    
    /// Reserved for future extensions
    pub reserved: [u8; 32],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * ExecutionType enum defines how proposal execution is handled
 */
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Eq, Copy)]
pub enum ExecutionType {
    /// Instructions execute independently
    Independent,
    
    /// All instructions must execute or all fail
    All,
    
    /// Instructions execute in sequence
    Sequential,
}

/**
 * ProposalInstruction account holds an instruction to be executed for a proposal
 */
#[account]
#[derive(Debug)]
pub struct ProposalInstruction {
    /// Proposal this instruction belongs to
    pub proposal: Pubkey,
    
    /// Index of this instruction in the proposal
    pub instruction_index: u8,
    
    /// Program to execute this instruction
    pub program_id: Pubkey,
    
    /// Accounts required for this instruction
    pub accounts: Vec<ProposalAccountMeta>,
    
    /// Instruction data
    pub data: Vec<u8>,
    
    /// Whether this instruction has been executed
    pub executed: bool,
    
    /// When the instruction was executed
    pub execution_time: Option<i64>,
    
    /// Pubkey of the executor
    pub executor: Option<Pubkey>,
    
    /// Version of the instruction format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * TokenOwnerRecord account tracks governance token deposits and voting power
 */
#[account]
#[derive(Debug)]
pub struct TokenOwnerRecord {
    /// Governance realm
    pub realm: Pubkey,
    
    /// Governance token mint
    pub governing_token_mint: Pubkey,
    
    /// Owner of the governance tokens
    pub governing_token_owner: Pubkey,
    
    /// Token account where the deposit is stored
    pub governing_token_deposit_account: Pubkey,
    
    /// Total tokens deposited
    pub governing_token_deposit_amount: u64,
    
    /// Unreleased tokens (not locked in proposal votes)
    pub unrelinquished_votes_count: u32,
    
    /// Whether this is for a council token
    pub is_council_token: bool,
    
    /// Account that can vote on behalf of the owner
    pub governance_delegate: Option<Pubkey>,
    
    /// Amount of voting power currently delegated
    pub delegated_voting_power: u64,
    
    /// Total voting power from delegations
    pub received_delegated_voting_power: u64,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 32],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * VoteRecord account tracks a vote on a proposal
 */
#[account]
#[derive(Debug)]
pub struct VoteRecord {
    /// Proposal being voted on
    pub proposal: Pubkey,
    
    /// Token owner record of the voter
    pub token_owner_record: Pubkey,
    
    /// Vote choice
    pub vote: VoteKind,
    
    /// Voting weight
    pub vote_weight: u64,
    
    /// Whether this vote is relinquished
    pub is_relinquished: bool,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * SignatoryRecord account tracks required signatories for a proposal
 */
#[account]
#[derive(Debug)]
pub struct SignatoryRecord {
    /// Proposal requiring signature
    pub proposal: Pubkey,
    
    /// Signatory account
    pub signatory: Pubkey,
    
    /// Whether signatory has signed off
    pub signed_off: bool,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * CommentRecord account stores comments on proposals
 */
#[account]
#[derive(Debug)]
pub struct CommentRecord {
    /// Proposal being commented on
    pub proposal: Pubkey,
    
    /// Author of the comment
    pub author: Pubkey,
    
    /// Comment text
    pub comment: [u8; MAX_COMMENT_LENGTH],
    
    /// Comment this is replying to
    pub reply_to: Option<u64>,
    
    /// Unique identifier for this comment
    pub comment_id: u64,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * EmergencyAuthorityRecord account tracks special emergency powers
 */
#[account]
#[derive(Debug)]
pub struct EmergencyAuthorityRecord {
    /// Governance realm
    pub realm: Pubkey,
    
    /// Authority pubkey
    pub authority: Pubkey,
    
    /// Allowed actions
    pub allowed_actions: EmergencyActions,
    
    /// When this authority expires
    pub expiry_time: i64,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

/**
 * TokenDistribution account tracks governance token distributions
 */
#[account]
#[derive(Debug)]
pub struct TokenDistribution {
    /// Governance realm
    pub realm: Pubkey,
    
    /// Token mint being distributed
    pub token_mint: Pubkey,
    
    /// Authority controlling the distribution
    pub authority: Pubkey,
    
    /// Type of distribution
    pub distribution_type: DistributionType,
    
    /// Recipients and amounts
    pub recipients: Vec<TokenRecipient>,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Version of this account format
    pub version: u8,
    
    /// Reserved for future extensions
    pub reserved: [u8; 8],
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

// Implementation blocks for PDA derivation and space calculation

impl Realm {
    /// Get the realm name as a string
    pub fn get_name(&self) -> String {
        let mut name_bytes = self.name.to_vec();
        let null_index = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_bytes.len());
        name_bytes.truncate(null_index);
        String::from_utf8(name_bytes).unwrap_or_else(|_| "Invalid UTF-8".to_string())
    }
    
    /// Calculate realm PDA
    pub fn find_address(
        program_id: &Pubkey,
        name: &str,
        community_token_mint: &Pubkey,
    ) -> (Pubkey, u8) {
        Pubkey::find_program_address(
            &[
                b"realm",
                name.as_bytes(),
                community_token_mint.as_ref(),
            ],
            program_id,
        )
    }
    
    /// Calculate space required for account
    pub fn get_space() -> usize {
        // Discriminator
        8 +
        // Optional authority pubkey
        1 + 32 +
        // Community token mint
        32 +
        // Optional council token mint
        1 + 32 +
        // Name
        MAX_NAME_LENGTH +
        // Config (approximate serialized size)
        200 +
        // Proposal count
        8 +
        // Created at timestamp
        8 +
        // Version
        1 +
        // Is active flag
        1 +
        // Reserved space
        64 +
        // Bump seed
        1
    }
}

impl Proposal {
    /// Get the proposal title as a string
    pub fn get_title(&self) -> String {
        let mut title_bytes = self.name.to_vec();
        let null_index = title_bytes.iter().position(|&b| b == 0).unwrap_or(title_bytes.len());
        title_bytes.truncate(null_index);
        String::from_utf8(title_bytes).unwrap_or_else(|_| "Invalid UTF-8".to_string())
    }
    
    /// Get the proposal description as a string
    pub fn get_description(&self) -> String {
        let mut desc_bytes = self.description.to_vec();
        let null_index = desc_bytes.iter().position(|&b| b == 0).unwrap_or(desc_bytes.len());
        desc_bytes.truncate(null_index);
        String::from_utf8(desc_bytes).unwrap_or_else(|_| "Invalid UTF-8".to_string())
    }
    
    /// Check if the proposal is in a final state
    pub fn is_final_state(&self) -> bool {
        matches!(
            self.state,
            ProposalStatus::Completed | 
            ProposalStatus::Cancelled | 
            ProposalStatus::Defeated |
            ProposalStatus::Vetoed |
            ProposalStatus::Expired
        )
    }
    
    /// Check if voting is active
    pub fn is_voting_active(&self) -> bool {
        if self.state != ProposalStatus::Voting {
            return false;
        }
        
        let clock = Clock::get().unwrap();
        let current_time = clock.unix_timestamp;
        
        current_time >= self.voting_start_time && current_time <= self.voting_end_time
    }
    
    /// Calculate proposal PDA
    pub fn find_address(
        program_id: &Pubkey,
        realm: &Pubkey,
        governing_token_mint: &Pubkey,
        proposal_index: u64,
    ) ->