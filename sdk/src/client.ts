/**
 * @file Minos-AI Client implementation
 * @module @minos-ai/sdk/client
 * @description Provides the main client class for interacting with the Minos-AI platform
 */

import { Connection, PublicKey, Commitment, Transaction, SendOptions, Keypair } from '@solana/web3.js';
import { Program, Provider, AnchorProvider, Idl, BN } from '@project-serum/anchor';
import EventEmitter from 'eventemitter3';
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import WebSocket from 'isomorphic-ws';
import { Buffer } from 'buffer';
import retry from 'retry';
import pRetry from 'p-retry';
import pTimeout from 'p-timeout';
import Debug from 'debug';

// Import utility functions and types
import { VaultClient } from './vault';
import { AgentClient } from './agent';
import {
  MinosClientConfig,
  NetworkEnvironment,
  ProgramIds,
  EventType,
  ErrorCode,
  MinosSdkError,
  ApiResponse,
  TransactionResult,
  ApiRequestOptions,
  TransactionOptions,
  WebhookEventType,
  WebhookPayload,
} from './types';

// Import constants
import {
  DEFAULT_PROGRAM_IDS,
  DEFAULT_API_ENDPOINT,
  DEFAULT_COMMITMENT,
  DEFAULT_TIMEOUT,
  DEFAULT_MAX_RETRIES,
  SDK_VERSION,
} from './constants';

// Create debug logger
const debug = Debug('minos-ai:client');

/**
 * Main client class for interacting with the Minos-AI platform
 */
export class MinosClient extends EventEmitter {
  /** Solana connection instance */
  public readonly connection: Connection;
  
  /** Anchor provider */
  public readonly provider: Provider;
  
  /** HTTP client for API requests */
  private readonly httpClient: AxiosInstance;
  
  /** WebSocket connection for real-time updates */
  private webSocket: WebSocket | null;
  
  /** Network environment */
  public readonly environment: NetworkEnvironment;
  
  /** Program IDs */
  public readonly programIds: ProgramIds;
  
  /** API endpoint */
  public readonly apiEndpoint: string;
  
  /** Request timeout */
  public readonly timeout: number;
  
  /** Whether to enable debug mode */
  public readonly debug: boolean;
  
  /** Maximum number of retries for failed operations */
  public readonly maxRetries: number;

  /** Vault client instance */
  public readonly vault: VaultClient;
  
  /** Agent client instance */
  public readonly agent: AgentClient;
  
  /** WebSocket endpoint */
  private readonly wsEndpoint?: string;
  
  /** Whether the client is connected via WebSocket */
  private isConnected: boolean = false;
  
  /** WebSocket heartbeat interval */
  private heartbeatInterval?: NodeJS.Timeout;
  
  /** WebSocket reconnect attempts */
  private reconnectAttempts: number = 0;
  
  /** Maximum WebSocket reconnect attempts */
  private readonly maxReconnectAttempts: number = 10;
  
  /** WebSocket reconnect delay */
  private readonly reconnectDelay: number = 1000;
  
  /** WebSocket connection timeout */
  private readonly wsConnectionTimeout: number = 30000;
  
  /** Program cache */
  private programCache: Map<string, Program> = new Map();
  
  /** Whether the client is initialized */
  private initialized: boolean = false;

  /**
   * Creates a new MinosClient instance
   * 
   * @param {MinosClientConfig} config - Client configuration
   */
  constructor(config: MinosClientConfig) {
    super();
    
    // Initialize from config
    this.connection = config.connection;
    this.provider = config.provider || new AnchorProvider(
      config.connection,
      // @ts-ignore - AnchorProvider expects a wallet but we might not have one
      null,
      { commitment: config.commitment || DEFAULT_COMMITMENT }
    );
    this.environment = config.environment || NetworkEnvironment.MAINNET;
    this.programIds = config.programIds || DEFAULT_PROGRAM_IDS[this.environment];
    this.apiEndpoint = config.apiEndpoint || DEFAULT_API_ENDPOINT;
    this.timeout = config.timeout || DEFAULT_TIMEOUT;
    this.debug = config.debug || false;
    this.maxRetries = config.maxRetries || DEFAULT_MAX_RETRIES;
    this.wsEndpoint = config.wsEndpoint;
    
    // Initialize HTTP client
    this.httpClient = axios.create({
      baseURL: this.apiEndpoint,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `MinosSDK/${SDK_VERSION}`,
        ...config.headers,
      },
    });

    // Initialize clients for submodules
    this.vault = new VaultClient(this);
    this.agent = new AgentClient(this);
    
    // Initialize WebSocket if endpoint is provided
    this.webSocket = null;
    if (this.wsEndpoint) {
      this.connectWebSocket();
    }
    
    // Set up debug mode
    if (this.debug) {
      Debug.enable('minos-ai:*');
    }
    
    // Log initialization
    debug('MinosClient initialized with environment: %s', this.environment);
    
    this.initialized = true;
  }

  /**
   * Initializes the client - loads program IDLs and verifies connection
   * 
   * @returns {Promise<void>}
   */
  public async initialize(): Promise<void> {
    if (!this.initialized) {
      // Verify connection
      await this.verifyConnection();
      
      // Load program IDLs
      await this.loadPrograms();
      
      this.initialized = true;
      debug('MinosClient fully initialized');
    }
  }

  /**
   * Verifies the Solana connection is working
   * 
   * @returns {Promise<boolean>} True if connection is working
   * @throws {MinosSdkError} If connection verification fails
   */
  public async verifyConnection(): Promise<boolean> {
    try {
      const version = await this.connection.getVersion();
      debug('Connected to Solana %s', version['solana-core']);
      return true;
    } catch (error) {
      const sdkError = new MinosSdkError(
        'Failed to connect to Solana cluster',
        ErrorCode.CONNECTION_ERROR,
        error instanceof Error ? error : undefined
      );
      this.emit(EventType.ERROR, sdkError);
      throw sdkError;
    }
  }

  /**
   * Loads all program IDLs
   * 
   * @returns {Promise<void>}
   */
  private async loadPrograms(): Promise<void> {
    try {
      // Load vault program
      await this.getProgram(this.programIds.vaultProgramId);
      
      // Load AI agent program
      await this.getProgram(this.programIds.aiAgentProgramId);
      
      // Load governance program
      await this.getProgram(this.programIds.governanceProgramId);
      
      debug('All programs loaded successfully');
    } catch (error) {
      const sdkError = new MinosSdkError(
        'Failed to load program IDLs',
        ErrorCode.PROGRAM_ERROR,
        error instanceof Error ? error : undefined
      );
      this.emit(EventType.ERROR, sdkError);
      throw sdkError;
    }
  }

  /**
   * Gets the program instance for a given program ID
   * 
   * @param {PublicKey} programId - Program ID
   * @returns {Promise<Program>} Program instance
   * @throws {MinosSdkError} If program loading fails
   */
  public async getProgram(programId: PublicKey): Promise<Program> {
    const programIdStr = programId.toString();
    
    // Check cache first
    if (this.programCache.has(programIdStr)) {
      return this.programCache.get(programIdStr)!;
    }
    
    try {
      // Fetch program IDL from on-chain account
      const idl = await Program.fetchIdl(programId, this.provider);
      
      if (!idl) {
        throw new Error(`IDL not found for program: ${programIdStr}`);
      }
      
      // Create program instance
      const program = new Program(idl, programId, this.provider);
      
      // Cache the program
      this.programCache.set(programIdStr, program);
      
      debug('Loaded program: %s', programIdStr);
      
      return program;
    } catch (error) {
      const sdkError = new MinosSdkError(
        `Failed to load program: ${programIdStr}`,
        ErrorCode.PROGRAM_ERROR,
        error instanceof Error ? error : undefined
      );
      this.emit(EventType.ERROR, sdkError);
      throw sdkError;
    }
  }

  /**
   * Connects to the WebSocket server for real-time updates
   * 
   * @returns {Promise<boolean>} True if connection succeeds
   */
  public async connectWebSocket(): Promise<boolean> {
    if (!this.wsEndpoint) {
      debug('WebSocket endpoint not provided, skipping connection');
      return false;
    }
    
    if (this.webSocket && this.isConnected) {
      debug('WebSocket already connected');
      return true;
    }
    
    return new Promise((resolve, reject) => {
      try {
        // Create new WebSocket connection
        this.webSocket = new WebSocket(this.wsEndpoint!);
        
        // Set up connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.webSocket && this.webSocket.readyState !== WebSocket.OPEN) {
            this.webSocket.close();
            const error = new MinosSdkError(
              'WebSocket connection timeout',
              ErrorCode.TIMEOUT_ERROR
            );
            this.emit(EventType.ERROR, error);
            reject(error);
          }
        }, this.wsConnectionTimeout);
        
        // Set up event handlers
        this.webSocket.onopen = () => {
          clearTimeout(connectionTimeout);
          this.isConnected = true;
          this.reconnectAttempts = 0;
          
          // Start heartbeat
          this.startHeartbeat();
          
          // Authenticate
          this.authenticateWebSocket();
          
          debug('WebSocket connected');
          this.emit(EventType.CONNECTED);
          resolve(true);
        };
        
        this.webSocket.onclose = () => {
          this.isConnected = false;
          this.stopHeartbeat();
          
          debug('WebSocket disconnected');
          this.emit(EventType.DISCONNECTED);
          
          // Attempt reconnect
          this.attemptReconnect();
          
          resolve(false);
        };
        
        this.webSocket.onerror = (error) => {
          const sdkError = new MinosSdkError(
            'WebSocket connection error',
            ErrorCode.CONNECTION_ERROR,
            error instanceof Error ? error : undefined
          );
          this.emit(EventType.ERROR, sdkError);
          
          // Will trigger onclose
          if (this.webSocket) {
            this.webSocket.close();
          }
        };
        
        this.webSocket.onmessage = (event) => {
          this.handleWebSocketMessage(event);
        };
      } catch (error) {
        const sdkError = new MinosSdkError(
          'Failed to initialize WebSocket connection',
          ErrorCode.CONNECTION_ERROR,
          error instanceof Error ? error : undefined
        );
        this.emit(EventType.ERROR, sdkError);
        reject(sdkError);
      }
    });
  }

  /**
   * Authenticates the WebSocket connection
   * 
   * @private
   */
  private authenticateWebSocket(): void {
    if (!this.webSocket || !this.isConnected) {
      return;
    }
    
    // Prepare authentication payload
    const authPayload = {
      action: 'authenticate',
      // Add authentication data if needed
    };
    
    // Send authentication message
    this.webSocket.send(JSON.stringify(authPayload));
    
    debug('WebSocket authentication sent');
  }

  /**
   * Handles incoming WebSocket messages
   * 
   * @param {WebSocket.MessageEvent} event - WebSocket message event
   * @private
   */
  private handleWebSocketMessage(event: WebSocket.MessageEvent): void {
    try {
      const message = JSON.parse(event.data as string);
      
      // Handle heartbeat response
      if (message.type === 'pong') {
        debug('Received heartbeat response');
        return;
      }
      
      // Handle event messages
      if (message.event) {
        this.handleWebhookEvent(message);
      }
      
      // Handle errors
      if (message.error) {
        const error = new MinosSdkError(
          message.error.message || 'WebSocket error',
          message.error.code || ErrorCode.UNKNOWN_ERROR
        );
        this.emit(EventType.ERROR, error);
      }
    } catch (error) {
      debug('Failed to parse WebSocket message: %o', error);
    }
  }

  /**
   * Handles webhook events from WebSocket
   * 
   * @param {WebhookPayload<any>} payload - Webhook payload
   * @private
   */
  private handleWebhookEvent(payload: WebhookPayload<any>): void {
    debug('Received webhook event: %s', payload.event);
    
    // Map webhook events to SDK event types
    switch (payload.event) {
      case WebhookEventType.VAULT_CREATED:
        this.emit(EventType.VAULT_CREATED, payload.data);
        break;
      case WebhookEventType.VAULT_UPDATED:
        this.emit(EventType.VAULT_UPDATED, payload.data);
        break;
      case WebhookEventType.AGENT_CREATED:
        this.emit(EventType.AGENT_CREATED, payload.data);
        break;
      case WebhookEventType.AGENT_UPDATED:
        this.emit(EventType.AGENT_UPDATED, payload.data);
        break;
      case WebhookEventType.DEPOSIT_COMPLETED:
        this.emit(EventType.DEPOSIT_COMPLETED, payload.data);
        break;
      case WebhookEventType.WITHDRAWAL_COMPLETED:
        this.emit(EventType.WITHDRAWAL_COMPLETED, payload.data);
        break;
      case WebhookEventType.ORDER_CREATED:
        this.emit(EventType.ORDER_CREATED, payload.data);
        break;
      case WebhookEventType.ORDER_UPDATED:
      case WebhookEventType.ORDER_FILLED:
        this.emit(EventType.ORDER_UPDATED, payload.data);
        break;
      case WebhookEventType.TRADE_EXECUTED:
        this.emit(EventType.TRADE_EXECUTED, payload.data);
        break;
      case WebhookEventType.PERFORMANCE_UPDATED:
        this.emit(EventType.PERFORMANCE_UPDATED, payload.data);
        break;
      default:
        // Emit generic event for unhandled types
        this.emit(payload.event, payload.data);
    }
  }

  /**
   * Starts the WebSocket heartbeat
   * 
   * @private
   */
  private startHeartbeat(): void {
    // Clear any existing interval
    this.stopHeartbeat();
    
    // Set up new heartbeat interval (every 30 seconds)
    this.heartbeatInterval = setInterval(() => {
      if (this.webSocket && this.isConnected) {
        const heartbeatMessage = { type: 'ping', timestamp: Date.now() };
        this.webSocket.send(JSON.stringify(heartbeatMessage));
        debug('Sent heartbeat');
      }
    }, 30000);
  }

  /**
   * Stops the WebSocket heartbeat
   * 
   * @private
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = undefined;
    }
  }

  /**
   * Attempts to reconnect to the WebSocket server
   * 
   * @private
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      debug('Maximum reconnect attempts reached');
      return;
    }
    
    // Exponential backoff
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts);
    this.reconnectAttempts++;
    
    debug('Attempting to reconnect in %d ms (attempt %d/%d)', 
      delay, this.reconnectAttempts, this.maxReconnectAttempts);
    
    setTimeout(() => {
      this.connectWebSocket().catch((error) => {
        debug('Reconnect attempt failed: %o', error);
      });
    }, delay);
  }

  /**
   * Disconnects from the WebSocket server
   */
  public disconnectWebSocket(): void {
    this.stopHeartbeat();
    
    if (this.webSocket) {
      this.webSocket.close();
      this.webSocket = null;
    }
    
    this.isConnected = false;
    debug('WebSocket disconnected');
  }

  /**
   * Makes an API request to the backend service
   * 
   * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
   * @param {string} endpoint - API endpoint
   * @param {any} data - Request data (for POST/PUT)
   * @param {ApiRequestOptions} options - Request options
   * @returns {Promise<T>} Response data
   * @throws {MinosSdkError} If the request fails
   */
  public async request<T = any>(
    method: string,
    endpoint: string,
    data?: any,
    options: ApiRequestOptions = {}
  ): Promise<T> {
    const requestOptions: AxiosRequestConfig = {
      method,
      url: endpoint,
      timeout: options.timeout || this.timeout,
      headers: {
        ...options.headers,
      },
    };

    // Add request data for POST/PUT
    if (data && (method.toUpperCase() === 'POST' || method.toUpperCase() === 'PUT')) {
      requestOptions.data = data;
    }
    
    // Add query parameters for GET/DELETE
    if (data && (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE')) {
      requestOptions.params = data;
    }
    
    // Add API version header if provided
    if (options.apiVersion) {
      requestOptions.headers!['X-API-Version'] = options.apiVersion;
    }
    
    // Add auth token if needed
    if (options.withAuth) {
      // Implement authentication logic here
      // For example, adding a JWT token
    }
    
    // Add user context if needed
    if (options.includeUserContext) {
      requestOptions.headers!['X-User-Context'] = 'true';
    }

    // Execute request with retries
    try {
      const response = await pRetry(
        async () => {
          // Execute request with timeout
          const result = await pTimeout(
            this.httpClient.request<ApiResponse<T>>(requestOptions),
            options.timeout || this.timeout,
            `Request to ${endpoint} timed out after ${options.timeout || this.timeout}ms`
          );
          
          // Check if the response indicates an error
          if (!result.data.success && result.data.error) {
            const error = new MinosSdkError(
              result.data.error,
              result.data.errorCode as ErrorCode || ErrorCode.API_ERROR
            );
            throw error;
          }
          
          return result.data.data as T;
        },
        {
          retries: this.maxRetries,
          onFailedAttempt: (error) => {
            debug(
              'API request to %s failed (attempt %d/%d): %s',
              endpoint,
              error.attemptNumber,
              this.maxRetries + 1,
              error.message
            );
          },
        }
      );

      return response;
    } catch (error) {
      let sdkError: MinosSdkError;
      
      if (error instanceof MinosSdkError) {
        sdkError = error;
      } else {
        sdkError = new MinosSdkError(
          `API request to ${endpoint} failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
      
      this.emit(EventType.ERROR, sdkError);
      throw sdkError;
    }
  }

  /**
   * Sends and confirms a transaction
   * 
   * @param {Transaction} transaction - Transaction to send
   * @param {Keypair[]} signers - Transaction signers
   * @param {TransactionOptions} options - Transaction options
   * @returns {Promise<TransactionResult>} Transaction result
   * @throws {MinosSdkError} If the transaction fails
   */
  public async sendAndConfirmTransaction(
    transaction: Transaction,
    signers: Keypair[] = [],
    options: TransactionOptions = {}
  ): Promise<TransactionResult> {
    try {
      // Clone the transaction to avoid modifying the original
      const tx = Transaction.from(transaction.serialize());
      
      // Set recent blockhash if not already set
      if (!tx.recentBlockhash) {
        const { blockhash } = await this.connection.getLatestBlockhash(
          options.commitment || DEFAULT_COMMITMENT
        );
        tx.recentBlockhash = blockhash;
      }
      
      // Set fee payer if not already set
      if (options.feePayer) {
        tx.feePayer = options.feePayer;
      } else if (this.provider instanceof AnchorProvider && this.provider.wallet) {
        tx.feePayer = this.provider.wallet.publicKey;
      }
      
      // Add additional signers if provided
      if (options.additionalSigners) {
        signers = [...signers, ...options.additionalSigners];
      }
      
      // Partial sign the transaction with provided signers
      if (signers.length > 0) {
        tx.partialSign(...signers);
      }
      
      // Sign with provider wallet if available
      if (this.provider instanceof AnchorProvider && this.provider.wallet) {
        await this.provider.wallet.signTransaction(tx);
      }
      
      // Prepare send options
      const sendOptions: SendOptions = {
        skipPreflight: options.skipPreflight || false,
        preflightCommitment: options.commitment || DEFAULT_COMMITMENT,
      };
      
      // Send the transaction
      const signature = await this.connection.sendRawTransaction(
        tx.serialize(),
        sendOptions
      );
      
      debug('Transaction sent with signature: %s', signature);
      
      // Confirm transaction if requested
      if (options.waitForConfirmation !== false) {
        const confirmationTimeoutMs = options.confirmationTimeout || 60000;
        
        const confirmation = await this.connection.confirmTransaction(
          {
            signature,
            blockhash: tx.recentBlockhash!,
            lastValidBlockHeight: (await this.connection.getBlockHeight()) + 150,
          },
          options.commitment || DEFAULT_COMMITMENT
        );
        
        if (confirmation.value.err) {
          throw new Error(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
        }
        
        debug('Transaction confirmed');
        
        // Get transaction details
        const transactionDetails = await this.connection.getTransaction(
          signature,
          { commitment: options.commitment || DEFAULT_COMMITMENT }
        );
        
        return {
          signature,
          transaction: tx,
          blockTime: transactionDetails?.blockTime || undefined,
          slot: transactionDetails?.slot,
        };
      }
      
      return {
        signature,
        transaction: tx,
      };
    } catch (error) {
      let errorCode = ErrorCode.TRANSACTION_ERROR;
      
      // Determine more specific error code if possible
      if (error instanceof Error) {
        if (error.message.includes('timeout')) {
          errorCode = ErrorCode.TIMEOUT_ERROR;
        } else if (error.message.includes('insufficient funds')) {
          errorCode = ErrorCode.INSUFFICIENT_FUNDS;
        } else if (error.message.includes('account not found')) {
          errorCode = ErrorCode.ACCOUNT_NOT_FOUND;
        } else if (error.message.includes('blockhash')) {
          errorCode = ErrorCode.CONFIRMATION_ERROR;
        }
      }
      
      const sdkError = new MinosSdkError(
        `Transaction failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        errorCode,
        error instanceof Error ? error : undefined
      );
      
      this.emit(EventType.ERROR, sdkError);
      throw sdkError;
    }
  }

  /**
   * Gets the current network time (slot and UNIX timestamp)
   * 
   * @returns {Promise<{slot: number, timestamp: number}>} Current network time
   */
  public async getNetworkTime(): Promise<{ slot: number; timestamp: number }> {
    const slot = await this.connection.getSlot();
    const timestamp = await this.connection.getBlockTime(slot);
    
    return {
      slot,
      timestamp: timestamp || Math.floor(Date.now() / 1000),
    };
  }

  /**
   * Gets the fee for sending a transaction
   * 
   * @returns {Promise<number>} Fee in lamports
   */
  public async getTransactionFee(): Promise<number> {
    const { feeCalculator } = await this.connection.getRecentBlockhash();
    return feeCalculator.lamportsPerSignature;
  }

  /**
   * Checks if an account exists
   * 
   * @param {PublicKey} accountId - Account public key
   * @returns {Promise<boolean>} True if the account exists
   */
  public async accountExists(accountId: PublicKey): Promise<boolean> {
    const accountInfo = await this.connection.getAccountInfo(accountId);
    return accountInfo !== null;
  }

  /**
   * Disposes the client, closing all connections
   */
  public dispose(): void {
    this.disconnectWebSocket();
    this.removeAllListeners();
    this.programCache.clear();
    debug('Client disposed');
  }
}