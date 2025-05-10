/**
 * @file Types and interfaces for the Minos-AI SDK
 * @module @minos-ai/sdk/types
 * @description This file contains all type definitions used throughout the Minos-AI SDK
 */

import { PublicKey, Commitment, Connection, Transaction, Keypair } from '@solana/web3.js';
import { Program, Provider, AnchorProvider, IdlAccounts } from '@project-serum/anchor';
import BN from 'bn.js';

/**
 * Network environment options for the Minos-AI platform
 */
export enum NetworkEnvironment {
  MAINNET = 'mainnet-beta',
  DEVNET = 'devnet',
  TESTNET = 'testnet',
  LOCALNET = 'localnet'
}

/**
 * Configuration options for the Minos-AI client
 */
export interface MinosClientConfig {
  /** Solana connection instance */
  connection: Connection;
  /** Commitment level for transactions */
  commitment?: Commitment;
  /** API endpoint for backend services */
  apiEndpoint?: string;
  /** Network environment */
  environment?: NetworkEnvironment;
  /** Optional provider for signing transactions */
  provider?: Provider | AnchorProvider;
  /** Custom program IDs (override defaults) */
  programIds?: ProgramIds;
  /** Timeout in milliseconds for requests */
  timeout?: number;
  /** Whether to enable verbose logging */
  debug?: boolean;
  /** Custom HTTP headers for API requests */
  headers?: Record<string, string>;
  /** Maximum number of retries for failed operations */
  maxRetries?: number;
  /** WebSocket endpoint for real-time updates */
  wsEndpoint?: string;
}

/**
 * Program IDs for various Minos-AI smart contracts
 */
export interface ProgramIds {
  /** Vault program ID */
  vaultProgramId: PublicKey;
  /** AI Agent program ID */
  aiAgentProgramId: PublicKey;
  /** Governance program ID */
  governanceProgramId: PublicKey;
}

/**
 * Default program IDs for different environments
 */
export interface DefaultProgramIds {
  /** Default program IDs for mainnet */
  mainnet: ProgramIds;
  /** Default program IDs for devnet */
  devnet: ProgramIds;
  /** Default program IDs for testnet */
  testnet: ProgramIds;
  /** Default program IDs for localnet */
  localnet: ProgramIds;
}

/**
 * Asset types supported by the platform
 */
export enum AssetType {
  SOL = 'SOL',
  SPL_TOKEN = 'SPL_TOKEN',
  USDC = 'USDC',
  BTC = 'BTC',
  ETH = 'ETH'
}

/**
 * Risk levels for trading strategies
 */
export enum RiskLevel {
  CONSERVATIVE = 'CONSERVATIVE',
  MODERATE = 'MODERATE',
  AGGRESSIVE = 'AGGRESSIVE',
  CUSTOM = 'CUSTOM'
}

/**
 * Time horizons for investment strategies
 */
export enum TimeHorizon {
  SHORT_TERM = 'SHORT_TERM',
  MEDIUM_TERM = 'MEDIUM_TERM',
  LONG_TERM = 'LONG_TERM'
}

/**
 * Trade order types
 */
export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT',
  TRAILING_STOP = 'TRAILING_STOP'
}

/**
 * Trade order side
 */
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

/**
 * Status of a trade order
 */
export enum OrderStatus {
  PENDING = 'PENDING',
  OPEN = 'OPEN',
  FILLED = 'FILLED',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  CANCELED = 'CANCELED',
  EXPIRED = 'EXPIRED',
  REJECTED = 'REJECTED'
}

/**
 * Status of a vault
 */
export enum VaultStatus {
  INITIALIZED = 'INITIALIZED',
  ACTIVE = 'ACTIVE',
  PAUSED = 'PAUSED',
  CLOSED = 'CLOSED'
}

/**
 * AI model types used for strategies
 */
export enum AiModelType {
  ARIADNE = 'ARIADNE',
  ANDROGEUS = 'ANDROGEUS',
  DEUCALION = 'DEUCALION',
  CUSTOM = 'CUSTOM'
}

/**
 * Fee structure for vaults
 */
export interface FeeStructure {
  /** Management fee percentage (annual) */
  managementFee: number;
  /** Performance fee percentage */
  performanceFee: number;
  /** Withdrawal fee percentage */
  withdrawalFee: number;
}

/**
 * Vault creation parameters
 */
export interface CreateVaultParams {
  /** Name of the vault */
  name: string;
  /** Description of the vault */
  description?: string;
  /** Initial deposit amount */
  initialDeposit: BN;
  /** Asset type for the vault */
  assetType: AssetType;
  /** Token mint address (for SPL tokens) */
  tokenMint?: PublicKey;
  /** Risk level for the vault's strategy */
  riskLevel: RiskLevel;
  /** Time horizon for the investment strategy */
  timeHorizon: TimeHorizon;
  /** Fee structure */
  fees?: FeeStructure;
  /** AI model type to use */
  aiModelType: AiModelType;
  /** Custom strategy parameters */
  strategyParams?: Record<string, any>;
  /** Maximum capacity of the vault */
  maxCapacity?: BN;
  /** Minimum deposit amount */
  minDeposit?: BN;
  /** Whether the vault allows withdrawals */
  allowWithdrawals?: boolean;
  /** Whether the vault is public or private */
  isPublic?: boolean;
  /** Admin authority for the vault */
  authority?: PublicKey;
}

/**
 * Vault account data from the blockchain
 */
export interface VaultAccount {
  /** Unique identifier */
  id: PublicKey;
  /** Vault name */
  name: string;
  /** Vault authority */
  authority: PublicKey;
  /** Asset type */
  assetType: AssetType;
  /** Token mint (for SPL tokens) */
  tokenMint: PublicKey | null;
  /** Vault token account */
  tokenAccount: PublicKey;
  /** Vault status */
  status: VaultStatus;
  /** Total assets under management */
  aum: BN;
  /** Number of investors */
  investorCount: number;
  /** Creation timestamp */
  createdAt: BN;
  /** Last updated timestamp */
  updatedAt: BN;
  /** Performance metrics */
  performance: VaultPerformance;
  /** Associated AI agent */
  agentId: PublicKey | null;
  /** Risk level */
  riskLevel: RiskLevel;
  /** Fee structure */
  fees: FeeStructure;
}

/**
 * Performance metrics for a vault
 */
export interface VaultPerformance {
  /** Total return percentage */
  totalReturn: number;
  /** Annualized return percentage */
  annualizedReturn: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Daily returns (last 30 days) */
  dailyReturns: number[];
  /** Monthly returns (last 12 months) */
  monthlyReturns: number[];
}

/**
 * Deposit parameters for a vault
 */
export interface DepositParams {
  /** Vault ID */
  vaultId: PublicKey;
  /** Amount to deposit */
  amount: BN;
  /** Token account to deposit from (for SPL tokens) */
  fromTokenAccount?: PublicKey;
  /** Reference ID for the deposit */
  referenceId?: string;
}

/**
 * Withdrawal parameters for a vault
 */
export interface WithdrawParams {
  /** Vault ID */
  vaultId: PublicKey;
  /** Amount to withdraw */
  amount: BN;
  /** Token account to withdraw to (for SPL tokens) */
  toTokenAccount?: PublicKey;
  /** Reference ID for the withdrawal */
  referenceId?: string;
}

/**
 * AI agent creation parameters
 */
export interface CreateAgentParams {
  /** Name of the agent */
  name: string;
  /** Description of the agent */
  description?: string;
  /** AI model type */
  modelType: AiModelType;
  /** Strategy parameters */
  strategyParams: Record<string, any>;
  /** Risk level */
  riskLevel: RiskLevel;
  /** Time horizon */
  timeHorizon: TimeHorizon;
  /** Associated vault ID (optional) */
  vaultId?: PublicKey;
  /** Authority of the agent */
  authority?: PublicKey;
}

/**
 * AI agent account data from the blockchain
 */
export interface AgentAccount {
  /** Unique identifier */
  id: PublicKey;
  /** Agent name */
  name: string;
  /** Agent authority */
  authority: PublicKey;
  /** AI model type */
  modelType: AiModelType;
  /** Strategy parameters */
  strategyParams: Record<string, any>;
  /** Risk level */
  riskLevel: RiskLevel;
  /** Time horizon */
  timeHorizon: TimeHorizon;
  /** Associated vault ID */
  vaultId: PublicKey | null;
  /** Performance metrics */
  performance: AgentPerformance;
  /** Creation timestamp */
  createdAt: BN;
  /** Last updated timestamp */
  updatedAt: BN;
  /** Whether the agent is active */
  isActive: boolean;
  /** Last execution timestamp */
  lastExecutedAt: BN | null;
}

/**
 * Performance metrics for an AI agent
 */
export interface AgentPerformance {
  /** Win rate percentage */
  winRate: number;
  /** Total number of trades */
  totalTrades: number;
  /** Profit factor */
  profitFactor: number;
  /** Average return per trade */
  averageReturn: number;
  /** Maximum consecutive wins */
  maxConsecutiveWins: number;
  /** Maximum consecutive losses */
  maxConsecutiveLosses: number;
}

/**
 * Parameters for creating a trade order
 */
export interface CreateOrderParams {
  /** Vault ID */
  vaultId: PublicKey;
  /** Agent ID */
  agentId: PublicKey;
  /** Market ID */
  marketId: PublicKey;
  /** Order type */
  orderType: OrderType;
  /** Order side */
  side: OrderSide;
  /** Asset quantity */
  quantity: BN;
  /** Price (for limit orders) */
  price?: BN;
  /** Stop price (for stop orders) */
  stopPrice?: BN;
  /** Time in force (in seconds) */
  timeInForce?: number;
  /** Client order ID */
  clientOrderId?: string;
}

/**
 * Trade order account data from the blockchain
 */
export interface OrderAccount {
  /** Unique identifier */
  id: PublicKey;
  /** Vault ID */
  vaultId: PublicKey;
  /** Agent ID */
  agentId: PublicKey;
  /** Market ID */
  marketId: PublicKey;
  /** Order type */
  orderType: OrderType;
  /** Order side */
  side: OrderSide;
  /** Asset quantity */
  quantity: BN;
  /** Original quantity */
  originalQuantity: BN;
  /** Filled quantity */
  filledQuantity: BN;
  /** Price (for limit orders) */
  price: BN | null;
  /** Stop price (for stop orders) */
  stopPrice: BN | null;
  /** Order status */
  status: OrderStatus;
  /** Creation timestamp */
  createdAt: BN;
  /** Last updated timestamp */
  updatedAt: BN;
  /** Client order ID */
  clientOrderId: string | null;
}

/**
 * Parameters for querying vaults
 */
export interface VaultQueryParams {
  /** Filter by status */
  status?: VaultStatus;
  /** Filter by asset type */
  assetType?: AssetType;
  /** Filter by risk level */
  riskLevel?: RiskLevel;
  /** Filter by authority */
  authority?: PublicKey;
  /** Filter by AI model type */
  aiModelType?: AiModelType;
  /** Filter by minimum AUM */
  minAum?: BN;
  /** Filter by maximum AUM */
  maxAum?: BN;
  /** Filter by minimum performance */
  minPerformance?: number;
  /** Sort by field */
  sortBy?: 'createdAt' | 'aum' | 'performance';
  /** Sort direction */
  sortDirection?: 'asc' | 'desc';
  /** Page number */
  page?: number;
  /** Page size */
  limit?: number;
}

/**
 * Parameters for querying agents
 */
export interface AgentQueryParams {
  /** Filter by model type */
  modelType?: AiModelType;
  /** Filter by risk level */
  riskLevel?: RiskLevel;
  /** Filter by authority */
  authority?: PublicKey;
  /** Filter by vault ID */
  vaultId?: PublicKey;
  /** Filter by active status */
  isActive?: boolean;
  /** Filter by minimum win rate */
  minWinRate?: number;
  /** Sort by field */
  sortBy?: 'createdAt' | 'winRate' | 'totalTrades';
  /** Sort direction */
  sortDirection?: 'asc' | 'desc';
  /** Page number */
  page?: number;
  /** Page size */
  limit?: number;
}

/**
 * Parameters for querying orders
 */
export interface OrderQueryParams {
  /** Filter by vault ID */
  vaultId?: PublicKey;
  /** Filter by agent ID */
  agentId?: PublicKey;
  /** Filter by market ID */
  marketId?: PublicKey;
  /** Filter by order status */
  status?: OrderStatus;
  /** Filter by order type */
  orderType?: OrderType;
  /** Filter by order side */
  side?: OrderSide;
  /** Filter by start date */
  startDate?: Date | number;
  /** Filter by end date */
  endDate?: Date | number;
  /** Sort by field */
  sortBy?: 'createdAt' | 'price' | 'quantity';
  /** Sort direction */
  sortDirection?: 'asc' | 'desc';
  /** Page number */
  page?: number;
  /** Page size */
  limit?: number;
}

/**
 * Response structure for paginated queries
 */
export interface PaginatedResponse<T> {
  /** Array of items */
  items: T[];
  /** Total count of items */
  total: number;
  /** Current page */
  page: number;
  /** Page size */
  limit: number;
  /** Number of pages */
  pages: number;
}

/**
 * API response structure
 */
export interface ApiResponse<T> {
  /** Success status */
  success: boolean;
  /** Response data */
  data?: T;
  /** Error message */
  error?: string;
  /** Error code */
  errorCode?: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Transaction result
 */
export interface TransactionResult {
  /** Transaction signature */
  signature: string;
  /** Transaction object */
  transaction: Transaction;
  /** Block time */
  blockTime?: number;
  /** Slot */
  slot?: number;
}

/**
 * Event types for client event emitter
 */
export enum EventType {
  VAULT_CREATED = 'vaultCreated',
  VAULT_UPDATED = 'vaultUpdated',
  AGENT_CREATED = 'agentCreated',
  AGENT_UPDATED = 'agentUpdated',
  DEPOSIT_COMPLETED = 'depositCompleted',
  WITHDRAWAL_COMPLETED = 'withdrawalCompleted',
  ORDER_CREATED = 'orderCreated',
  ORDER_UPDATED = 'orderUpdated',
  TRADE_EXECUTED = 'tradeExecuted',
  PERFORMANCE_UPDATED = 'performanceUpdated',
  ERROR = 'error',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected'
}

/**
 * Error codes
 */
export enum ErrorCode {
  // Connection errors
  CONNECTION_ERROR = 'CONNECTION_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  NETWORK_ERROR = 'NETWORK_ERROR',
  
  // Transaction errors
  TRANSACTION_ERROR = 'TRANSACTION_ERROR',
  SIMULATION_ERROR = 'SIMULATION_ERROR',
  CONFIRMATION_ERROR = 'CONFIRMATION_ERROR',
  
  // Validation errors
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  INVALID_PARAMS = 'INVALID_PARAMS',
  INSUFFICIENT_FUNDS = 'INSUFFICIENT_FUNDS',
  
  // Account errors
  ACCOUNT_NOT_FOUND = 'ACCOUNT_NOT_FOUND',
  ACCOUNT_ALREADY_EXISTS = 'ACCOUNT_ALREADY_EXISTS',
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN',
  
  // API errors
  API_ERROR = 'API_ERROR',
  NOT_FOUND = 'NOT_FOUND',
  BAD_REQUEST = 'BAD_REQUEST',
  SERVER_ERROR = 'SERVER_ERROR',
  
  // Program errors
  PROGRAM_ERROR = 'PROGRAM_ERROR',
  INSTRUCTION_ERROR = 'INSTRUCTION_ERROR',
  
  // Other errors
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
  NOT_IMPLEMENTED = 'NOT_IMPLEMENTED',
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED'
}

/**
 * Custom error class for SDK errors
 */
export class MinosSdkError extends Error {
  /** Error code */
  code: ErrorCode;
  /** Original error */
  originalError?: Error;
  /** Additional context */
  context?: Record<string, any>;

  /**
   * Creates a new MinosSdkError
   * @param message Error message
   * @param code Error code
   * @param originalError Original error
   * @param context Additional context
   */
  constructor(
    message: string, 
    code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    originalError?: Error,
    context?: Record<string, any>
  ) {
    super(message);
    this.name = 'MinosSdkError';
    this.code = code;
    this.originalError = originalError;
    this.context = context;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, MinosSdkError);
    }
  }
}

/**
 * Webhook event types
 */
export enum WebhookEventType {
  VAULT_CREATED = 'vault.created',
  VAULT_UPDATED = 'vault.updated',
  VAULT_CLOSED = 'vault.closed',
  DEPOSIT_COMPLETED = 'deposit.completed',
  WITHDRAWAL_COMPLETED = 'withdrawal.completed',
  TRADE_EXECUTED = 'trade.executed',
  PERFORMANCE_UPDATED = 'performance.updated',
  AGENT_CREATED = 'agent.created',
  AGENT_UPDATED = 'agent.updated',
  ORDER_CREATED = 'order.created',
  ORDER_UPDATED = 'order.updated',
  ORDER_FILLED = 'order.filled'
}

/**
 * Webhook payload structure
 */
export interface WebhookPayload<T> {
  /** Event type */
  event: WebhookEventType;
  /** Event data */
  data: T;
  /** Timestamp */
  timestamp: number;
  /** Account ID */
  accountId: string;
  /** Environment */
  environment: NetworkEnvironment;
  /** Webhook ID */
  webhookId: string;
}

/**
 * Strategy parameters for different AI models
 */

/**
 * Ariadne model strategy parameters
 */
export interface AriadneStrategyParams {
  /** Time interval for analysis */
  timeInterval: '5m' | '15m' | '1h' | '4h' | '1d';
  /** Lookback period */
  lookbackPeriod: number;
  /** Technical indicators to use */
  indicators: string[];
  /** Risk allocation percentage */
  riskAllocation: number;
  /** Maximum drawdown percentage */
  maxDrawdown: number;
  /** Take profit percentage */
  takeProfit: number;
  /** Stop loss percentage */
  stopLoss: number;
  /** Trade sizing method */
  tradeSizingMethod: 'fixed' | 'percentage' | 'kelly' | 'volatility';
  /** Position sizing percentage */
  positionSizePercentage: number;
}

/**
 * Androgeus model strategy parameters
 */
export interface AndrogeusStrategyParams {
  /** Technical analysis parameters */
  technicalAnalysis: {
    /** Moving average periods */
    maPeriods: number[];
    /** RSI parameters */
    rsi: {
      period: number;
      overbought: number;
      oversold: number;
    };
    /** MACD parameters */
    macd: {
      fastPeriod: number;
      slowPeriod: number;
      signalPeriod: number;
    };
    /** Bollinger bands parameters */
    bollingerBands: {
      period: number;
      deviations: number;
    };
  };
  /** Risk management parameters */
  riskManagement: {
    /** Maximum position size */
    maxPositionSize: number;
    /** Maximum risk per trade */
    maxRiskPerTrade: number;
    /** Stop loss type */
    stopLossType: 'fixed' | 'atr' | 'volatility';
    /** Stop loss parameter */
    stopLossParam: number;
    /** Take profit type */
    takeProfitType: 'fixed' | 'risk-reward' | 'trailing';
    /** Take profit parameter */
    takeProfitParam: number;
  };
  /** Position management parameters */
  positionManagement: {
    /** Whether to allow partial exits */
    partialExits: boolean;
    /** Whether to scale in positions */
    scaleIn: boolean;
    /** Position sizing method */
    positionSizingMethod: 'fixed' | 'percentage' | 'kelly' | 'volatility';
  };
}

/**
 * Deucalion model strategy parameters
 */
export interface DeucalionStrategyParams {
  /** Social data sources */
  dataSources: ('twitter' | 'reddit' | 'news' | 'discord' | 'github')[];
  /** Sentiment analysis parameters */
  sentiment: {
    /** Minimum sentiment score for action */
    minSentimentScore: number;
    /** Lookback period */
    lookbackPeriod: number;
    /** Whether to use advanced NLP */
    useAdvancedNlp: boolean;
  };
  /** Market data parameters */
  marketData: {
    /** Whether to correlate with market data */
    correlateWithMarket: boolean;
    /** Market data sources */
    sources: string[];
  };
  /** Copy trading parameters */
  copyTrading: {
    /** Whether to use copy trading */
    enabled: boolean;
    /** Minimum trader score */
    minTraderScore: number;
    /** Maximum number of traders to follow */
    maxTradersToFollow: number;
    /** Allocation weight method */
    allocationWeightMethod: 'equal' | 'performance' | 'customized';
  };
  /** Risk management parameters */
  riskManagement: {
    /** Maximum allocation per asset */
    maxAllocationPerAsset: number;
    /** Maximum allocation per trader */
    maxAllocationPerTrader: number;
    /** Position sizing method */
    positionSizingMethod: 'fixed' | 'percentage' | 'proportional';
  };
}

/**
 * Custom strategy parameters
 */
export interface CustomStrategyParams {
  /** Strategy definition in JSON format */
  strategyDefinition: Record<string, any>;
  /** Custom parameters */
  parameters: Record<string, any>;
  /** Backtesting results */
  backtestingResults?: {
    /** Sharpe ratio */
    sharpeRatio: number;
    /** Maximum drawdown */
    maxDrawdown: number;
    /** Total return */
    totalReturn: number;
    /** Win rate */
    winRate: number;
  };
  /** Risk management settings */
  riskManagement: Record<string, any>;
}

/**
 * Union type for all strategy parameters
 */
export type StrategyParams = 
  | AriadneStrategyParams 
  | AndrogeusStrategyParams 
  | DeucalionStrategyParams 
  | CustomStrategyParams;

/**
 * Market data types
 */
export interface MarketData {
  /** Market ID */
  marketId: string;
  /** Symbol */
  symbol: string;
  /** Base asset */
  baseAsset: string;
  /** Quote asset */
  quoteAsset: string;
  /** Current price */
  price: number;
  /** 24h price change percentage */
  priceChangePercent: number;
  /** 24h high */
  high24h: number;
  /** 24h low */
  low24h: number;
  /** 24h volume */
  volume24h: number;
  /** Timestamp */
  timestamp: number;
}

/**
 * Candlestick data
 */
export interface Candlestick {
  /** Open time */
  openTime: number;
  /** Open price */
  open: number;
  /** High price */
  high: number;
  /** Low price */
  low: number;
  /** Close price */
  close: number;
  /** Volume */
  volume: number;
  /** Close time */
  closeTime: number;
  /** Quote asset volume */
  quoteAssetVolume: number;
  /** Number of trades */
  trades: number;
}

/**
 * Trade data
 */
export interface Trade {
  /** Trade ID */
  id: string;
  /** Order ID */
  orderId: string;
  /** Symbol */
  symbol: string;
  /** Price */
  price: number;
  /** Quantity */
  quantity: number;
  /** Quote quantity */
  quoteQuantity: number;
  /** Commission */
  commission: number;
  /** Commission asset */
  commissionAsset: string;
  /** Trade time */
  time: number;
  /** Whether the trade is buyer maker */
  isBuyerMaker: boolean;
  /** Whether the trade is best match */
  isBestMatch: boolean;
}

/**
 * User account data
 */
export interface UserAccount {
  /** User ID */
  id: string;
  /** Public key */
  publicKey: string;
  /** Email */
  email?: string;
  /** Username */
  username?: string;
  /** Created at timestamp */
  createdAt: number;
  /** Last login timestamp */
  lastLoginAt?: number;
  /** User settings */
  settings?: Record<string, any>;
  /** Permissions */
  permissions?: string[];
}

/**
 * Dashboard data
 */
export interface DashboardData {
  /** Total assets under management */
  totalAum: BN;
  /** Total return */
  totalReturn: number;
  /** Number of active vaults */
  activeVaults: number;
  /** Number of active agents */
  activeAgents: number;
  /** Performance history */
  performanceHistory: Array<{
    /** Timestamp */
    timestamp: number;
    /** Value */
    value: number;
  }>;
  /** Asset allocation */
  assetAllocation: Record<string, number>;
  /** Recent activities */
  recentActivities: Array<{
    /** Activity type */
    type: string;
    /** Description */
    description: string;
    /** Timestamp */
    timestamp: number;
    /** Reference ID */
    referenceId?: string;
  }>;
}

/**
 * Options for transaction handling
 */
export interface TransactionOptions {
  /** Skip preflight check */
  skipPreflight?: boolean;
  /** Maximum retries */
  maxRetries?: number;
  /** Commitment level */
  commitment?: Commitment;
  /** Whether to wait for confirmation */
  waitForConfirmation?: boolean;
  /** Confirmation timeout (in milliseconds) */
  confirmationTimeout?: number;
  /** Recent blockhash for transaction */
  recentBlockhash?: string;
  /** Fee payer */
  feePayer?: PublicKey;
  /** Additional signers */
  additionalSigners?: Keypair[];
}

/**
 * Options for API requests
 */
export interface ApiRequestOptions {
  /** Request timeout (in milliseconds) */
  timeout?: number;
  /** Whether to include authentication */
  withAuth?: boolean;
  /** Additional headers */
  headers?: Record<string, string>;
  /** API version */
  apiVersion?: string;
  /** Whether to include user context */
  includeUserContext?: boolean;
}

/**
 * Solana account with data
 */
export interface AccountWithData<T> {
  /** Public key */
  publicKey: PublicKey;
  /** Account data */
  data: T;
}

/**
 * Export all IDL account types for smart contracts
 */
export namespace VaultIDL {
  export type VaultAccount = IdlAccounts<any>['vault'];
  export type InvestorAccount = IdlAccounts<any>['investor'];
  export type TransactionAccount = IdlAccounts<any>['transaction'];
}

export namespace AgentIDL {
  export type AgentAccount = IdlAccounts<any>['agent'];
  export type StrategyAccount = IdlAccounts<any>['strategy'];
  export type ModelConfigAccount = IdlAccounts<any>['modelConfig'];
}

export namespace GovernanceIDL {
  export type GovernanceAccount = IdlAccounts<any>['governance'];
  export type ProposalAccount = IdlAccounts<any>['proposal'];
  export type VoteAccount = IdlAccounts<any>['vote'];
}