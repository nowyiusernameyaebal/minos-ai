/**
 * @file Agent client implementation
 * @module @minos-ai/sdk/agent
 * @description Provides functionality for interacting with Minos-AI trading agents
 */

import {
    PublicKey,
    Transaction,
    SystemProgram,
    SYSVAR_RENT_PUBKEY,
    TransactionInstruction,
    Keypair,
  } from '@solana/web3.js';
  import { 
    Program, 
    BN, 
    utils, 
    web3,
    ProgramAccount,
  } from '@project-serum/anchor';
  import { Buffer } from 'buffer';
  
  // Import client base and types
  import { MinosClient } from './client';
  import {
    AgentAccount,
    CreateAgentParams,
    AgentQueryParams,
    RiskLevel,
    TimeHorizon,
    AiModelType,
    PaginatedResponse,
    TransactionResult,
    TransactionOptions,
    ApiRequestOptions,
    ErrorCode,
    MinosSdkError,
    AgentPerformance,
    AccountWithData,
    ApiResponse,
    StrategyParams,
    AriadneStrategyParams,
    AndrogeusStrategyParams,
    DeucalionStrategyParams,
    CustomStrategyParams,
    CreateOrderParams,
    OrderAccount,
    OrderQueryParams,
    OrderType,
    OrderSide,
    OrderStatus,
  } from './types';
  
  // Import constants and utilities
  import {
    DEFAULT_COMMITMENT,
    AGENT_SEED,
    AGENT_AUTHORITY_SEED,
    STRATEGY_SEED,
    ORDER_SEED,
  } from './constants';
  import { 
    validatePublicKey,
    serializeStrategyParams,
    deserializeStrategyParams,
  } from './utils';
  
  /**
   * AgentClient provides methods for interacting with AI trading agents in the Minos-AI platform
   */
  export class AgentClient {
    private readonly client: MinosClient;
  
    /**
     * Creates a new AgentClient instance
     * 
     * @param {MinosClient} client - MinosClient instance
     */
    constructor(client: MinosClient) {
      this.client = client;
    }
  
    /**
     * Creates a new AI trading agent
     * 
     * @param {CreateAgentParams} params - Agent creation parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<AgentAccount>} The created agent account
     * @throws {MinosSdkError} If agent creation fails
     * 
     * @example
     * ```typescript
     * const agent = await client.agent.createAgent({
     *   name: 'My AI Trading Agent',
     *   modelType: AiModelType.ARIADNE,
     *   strategyParams: {
     *     timeInterval: '1h',
     *     lookbackPeriod: 14,
     *     indicators: ['RSI', 'MACD', 'BB'],
     *     riskAllocation: 5,
     *     maxDrawdown: 10,
     *     takeProfit: 15,
     *     stopLoss: 5,
     *     tradeSizingMethod: 'percentage',
     *     positionSizePercentage: 10
     *   },
     *   riskLevel: RiskLevel.MODERATE,
     *   timeHorizon: TimeHorizon.MEDIUM_TERM,
     *   vaultId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * });
     * console.log('Created agent:', agent);
     * ```
     */
    public async createAgent(
      params: CreateAgentParams,
      options: TransactionOptions = {}
    ): Promise<AgentAccount> {
      try {
        // Validate parameters
        if (!params.name) {
          throw new MinosSdkError(
            'Agent name is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.modelType) {
          throw new MinosSdkError(
            'Model type is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.strategyParams) {
          throw new MinosSdkError(
            'Strategy parameters are required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.riskLevel) {
          throw new MinosSdkError(
            'Risk level is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.timeHorizon) {
          throw new MinosSdkError(
            'Time horizon is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the agent program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Generate a new keypair for the agent if not provided
        const agentKeypair = Keypair.generate();
        const agentId = agentKeypair.publicKey;
  
        // Derive the agent authority PDA
        const [agentAuthority, agentAuthorityBump] = await this.findAgentAuthorityAddress(
          agentId
        );
  
        // Derive the strategy account PDA
        const [strategyAccount, strategyBump] = await this.findStrategyAddress(
          agentId
        );
  
        // Serialize strategy parameters based on model type
        const serializedStrategyParams = serializeStrategyParams(
          params.modelType,
          params.strategyParams
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add create agent instruction
        const createAgentIx = await this.buildCreateAgentInstruction(
          program,
          {
            agentId,
            agentAuthority,
            creator: this.client.provider.publicKey,
            strategyAccount,
            name: params.name,
            description: params.description || '',
            modelType: params.modelType,
            riskLevel: params.riskLevel,
            timeHorizon: params.timeHorizon,
            vaultId: params.vaultId || null,
            strategyParams: serializedStrategyParams,
            authority: params.authority || this.client.provider.publicKey,
          }
        );
        tx.add(createAgentIx);
  
        // Send the transaction
        const signers = [agentKeypair];
        const result = await this.client.sendAndConfirmTransaction(tx, signers, options);
  
        // Fetch the created agent
        const agent = await this.getAgent(agentId);
        if (!agent) {
          throw new MinosSdkError(
            'Failed to fetch created agent',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return agent;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to create agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets an agent by its ID
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<AgentAccount | null>} The agent account or null if not found
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const agentId = new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU');
     * const agent = await client.agent.getAgent(agentId);
     * if (agent) {
     *   console.log('Found agent:', agent);
     * } else {
     *   console.log('Agent not found');
     * }
     * ```
     */
    public async getAgent(
      agentId: PublicKey | string,
      options: ApiRequestOptions = {}
    ): Promise<AgentAccount | null> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        // Attempt to fetch from the blockchain
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        try {
          const agentAccount = await program.account.agent.fetch(publicKey);
          
          // Also fetch strategy account to get complete data
          const [strategyAddress] = await this.findStrategyAddress(publicKey);
          const strategyAccount = await program.account.strategy.fetch(strategyAddress);
          
          return this.parseAgentAccount(publicKey, agentAccount, strategyAccount);
        } catch (error) {
          // If not found on-chain, try API
          try {
            return await this.client.request<AgentAccount>(
              'GET',
              `/v1/agents/${publicKey.toString()}`,
              undefined,
              options
            );
          } catch (apiError) {
            // If not found via API either, return null
            if (
              apiError instanceof MinosSdkError &&
              apiError.code === ErrorCode.NOT_FOUND
            ) {
              return null;
            }
            throw apiError;
          }
        }
      } catch (error) {
        if (
          error instanceof MinosSdkError &&
          error.code === ErrorCode.NOT_FOUND
        ) {
          return null;
        }
  
        throw new MinosSdkError(
          `Failed to get agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all agents matching the specified query parameters
     * 
     * @param {AgentQueryParams} params - Query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<AgentAccount>>} Paginated list of agents
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const agents = await client.agent.getAgents({
     *   modelType: AiModelType.ARIADNE,
     *   isActive: true,
     *   sortBy: 'winRate',
     *   sortDirection: 'desc',
     *   page: 1,
     *   limit: 10
     * });
     * console.log(`Found ${agents.total} agents, showing ${agents.items.length}`);
     * ```
     */
    public async getAgents(
      params: AgentQueryParams = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<AgentAccount>> {
      try {
        // Convert PublicKey to string
        const queryParams: Record<string, any> = { ...params };
        if (params.authority && params.authority instanceof PublicKey) {
          queryParams.authority = params.authority.toString();
        }
        if (params.vaultId && params.vaultId instanceof PublicKey) {
          queryParams.vaultId = params.vaultId.toString();
        }
  
        // Set default pagination
        if (!queryParams.page) {
          queryParams.page = 1;
        }
        if (!queryParams.limit) {
          queryParams.limit = 10;
        }
  
        return await this.client.request<PaginatedResponse<AgentAccount>>(
          'GET',
          '/v1/agents',
          queryParams,
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get agents: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all agents owned by the specified authority
     * 
     * @param {PublicKey | string} authority - Agent authority
     * @param {AgentQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<AgentAccount>>} Paginated list of agents
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const myAgents = await client.agent.getAgentsByAuthority(client.provider.publicKey);
     * console.log(`You own ${myAgents.total} agents`);
     * ```
     */
    public async getAgentsByAuthority(
      authority: PublicKey | string,
      params: Omit<AgentQueryParams, 'authority'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<AgentAccount>> {
      try {
        const publicKey = validatePublicKey(authority);
        return await this.getAgents({ ...params, authority: publicKey }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get agents by authority: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all agents associated with a vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {AgentQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<AgentAccount>>} Paginated list of agents
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const vaultAgents = await client.agent.getAgentsByVault(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log(`This vault has ${vaultAgents.total} agents`);
     * ```
     */
    public async getAgentsByVault(
      vaultId: PublicKey | string,
      params: Omit<AgentQueryParams, 'vaultId'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<AgentAccount>> {
      try {
        const publicKey = validatePublicKey(vaultId);
        return await this.getAgents({ ...params, vaultId: publicKey }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get agents by vault: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all agents using a specific AI model type
     * 
     * @param {AiModelType} modelType - AI model type
     * @param {AgentQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<AgentAccount>>} Paginated list of agents
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const ariadneAgents = await client.agent.getAgentsByModelType(AiModelType.ARIADNE);
     * console.log(`Found ${ariadneAgents.total} agents using Ariadne model`);
     * ```
     */
    public async getAgentsByModelType(
      modelType: AiModelType,
      params: Omit<AgentQueryParams, 'modelType'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<AgentAccount>> {
      try {
        return await this.getAgents({ ...params, modelType }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get agents by model type: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets the performance history of an agent
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {string} period - Time period ('1d', '1w', '1m', '3m', '1y', 'all')
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<AgentPerformance>} Agent performance data
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const performance = await client.agent.getAgentPerformance(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   '1m'
     * );
     * console.log('Agent performance:', performance);
     * ```
     */
    public async getAgentPerformance(
      agentId: PublicKey | string,
      period: '1d' | '1w' | '1m' | '3m' | '1y' | 'all' = '1m',
      options: ApiRequestOptions = {}
    ): Promise<AgentPerformance> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        return await this.client.request<AgentPerformance>(
          'GET',
          `/v1/agents/${publicKey.toString()}/performance`,
          { period },
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get agent performance: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Updates an agent's strategy parameters
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {StrategyParams} strategyParams - New strategy parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<AgentAccount>} Updated agent account
     * @throws {MinosSdkError} If the update fails
     * 
     * @example
     * ```typescript
     * const updatedAgent = await client.agent.updateStrategy(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   {
     *     timeInterval: '4h',
     *     lookbackPeriod: 21,
     *     indicators: ['RSI', 'MACD', 'BB', 'ATR'],
     *     riskAllocation: 3,
     *     maxDrawdown: 8,
     *     takeProfit: 20,
     *     stopLoss: 5,
     *     tradeSizingMethod: 'percentage',
     *     positionSizePercentage: 8
     *   }
     * );
     * console.log('Updated agent strategy:', updatedAgent);
     * ```
     */
    public async updateStrategy(
      agentId: PublicKey | string,
      strategyParams: StrategyParams,
      options: TransactionOptions = {}
    ): Promise<AgentAccount> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        // Get the agent
        const agent = await this.getAgent(publicKey);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can update the strategy',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Generate a new keypair for the order
        const orderKeypair = Keypair.generate();
        const orderId = orderKeypair.publicKey;
  
        // Get the agent authority PDA
        const [agentAuthority, agentAuthorityBump] = await this.findAgentAuthorityAddress(
          new PublicKey(params.agentId)
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add create order instruction
        const createOrderIx = await this.buildCreateOrderInstruction(
          program,
          {
            orderId,
            agentId: new PublicKey(params.agentId),
            agentAuthority,
            vaultId: new PublicKey(params.vaultId),
            marketId: new PublicKey(params.marketId),
            authority: this.client.provider.publicKey,
            orderType: params.orderType,
            side: params.side,
            quantity: params.quantity,
            price: params.price || null,
            stopPrice: params.stopPrice || null,
            timeInForce: params.timeInForce || 0,
            clientOrderId: params.clientOrderId || null,
            agentAuthorityBump,
          }
        );
        tx.add(createOrderIx);
  
        // Send the transaction
        const signers = [orderKeypair];
        const result = await this.client.sendAndConfirmTransaction(tx, signers, options);
  
        // Fetch the created order
        const order = await this.getOrder(orderId);
        if (!order) {
          throw new MinosSdkError(
            'Failed to fetch created order',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return order;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to create order: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets an order by its ID
     * 
     * @param {PublicKey | string} orderId - Order ID
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<OrderAccount | null>} The order account or null if not found
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const orderId = new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU');
     * const order = await client.agent.getOrder(orderId);
     * if (order) {
     *   console.log('Found order:', order);
     * } else {
     *   console.log('Order not found');
     * }
     * ```
     */
    public async getOrder(
      orderId: PublicKey | string,
      options: ApiRequestOptions = {}
    ): Promise<OrderAccount | null> {
      try {
        const publicKey = validatePublicKey(orderId);
  
        // Attempt to fetch from the blockchain
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        try {
          const orderAccount = await program.account.order.fetch(publicKey);
          return this.parseOrderAccount(publicKey, orderAccount);
        } catch (error) {
          // If not found on-chain, try API
          try {
            return await this.client.request<OrderAccount>(
              'GET',
              `/v1/orders/${publicKey.toString()}`,
              undefined,
              options
            );
          } catch (apiError) {
            // If not found via API either, return null
            if (
              apiError instanceof MinosSdkError &&
              apiError.code === ErrorCode.NOT_FOUND
            ) {
              return null;
            }
            throw apiError;
          }
        }
      } catch (error) {
        if (
          error instanceof MinosSdkError &&
          error.code === ErrorCode.NOT_FOUND
        ) {
          return null;
        }
  
        throw new MinosSdkError(
          `Failed to get order: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all orders matching the specified query parameters
     * 
     * @param {OrderQueryParams} params - Query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<OrderAccount>>} Paginated list of orders
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const orders = await client.agent.getOrders({
     *   agentId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   status: OrderStatus.OPEN,
     *   side: OrderSide.BUY,
     *   sortBy: 'createdAt',
     *   sortDirection: 'desc',
     *   page: 1,
     *   limit: 10
     * });
     * console.log(`Found ${orders.total} orders, showing ${orders.items.length}`);
     * ```
     */
    public async getOrders(
      params: OrderQueryParams = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<OrderAccount>> {
      try {
        // Convert PublicKey to string
        const queryParams: Record<string, any> = { ...params };
        if (params.vaultId && params.vaultId instanceof PublicKey) {
          queryParams.vaultId = params.vaultId.toString();
        }
        if (params.agentId && params.agentId instanceof PublicKey) {
          queryParams.agentId = params.agentId.toString();
        }
        if (params.marketId && params.marketId instanceof PublicKey) {
          queryParams.marketId = params.marketId.toString();
        }
  
        // Convert Date to timestamp if needed
        if (params.startDate) {
          if (params.startDate instanceof Date) {
            queryParams.startDate = Math.floor(params.startDate.getTime() / 1000);
          }
        }
        if (params.endDate) {
          if (params.endDate instanceof Date) {
            queryParams.endDate = Math.floor(params.endDate.getTime() / 1000);
          }
        }
  
        // Set default pagination
        if (!queryParams.page) {
          queryParams.page = 1;
        }
        if (!queryParams.limit) {
          queryParams.limit = 10;
        }
  
        return await this.client.request<PaginatedResponse<OrderAccount>>(
          'GET',
          '/v1/orders',
          queryParams,
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get orders: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all orders for a specific agent
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {OrderQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<OrderAccount>>} Paginated list of orders
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const agentOrders = await client.agent.getOrdersByAgent(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log(`This agent has ${agentOrders.total} orders`);
     * ```
     */
    public async getOrdersByAgent(
      agentId: PublicKey | string,
      params: Omit<OrderQueryParams, 'agentId'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<OrderAccount>> {
      try {
        const publicKey = validatePublicKey(agentId);
        return await this.getOrders({ ...params, agentId: publicKey }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get orders by agent: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all orders for a specific vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {OrderQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<OrderAccount>>} Paginated list of orders
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const vaultOrders = await client.agent.getOrdersByVault(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log(`This vault has ${vaultOrders.total} orders`);
     * ```
     */
    public async getOrdersByVault(
      vaultId: PublicKey | string,
      params: Omit<OrderQueryParams, 'vaultId'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<OrderAccount>> {
      try {
        const publicKey = validatePublicKey(vaultId);
        return await this.getOrders({ ...params, vaultId: publicKey }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get orders by vault: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Cancels an order
     * 
     * @param {PublicKey | string} orderId - Order ID
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TransactionResult>} Transaction result
     * @throws {MinosSdkError} If the cancellation fails
     * 
     * @example
     * ```typescript
     * const result = await client.agent.cancelOrder(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log('Order canceled:', result.signature);
     * ```
     */
    public async cancelOrder(
      orderId: PublicKey | string,
      options: TransactionOptions = {}
    ): Promise<TransactionResult> {
      try {
        const publicKey = validatePublicKey(orderId);
  
        // Get the order
        const order = await this.getOrder(publicKey);
        if (!order) {
          throw new MinosSdkError(
            `Order not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the order is already canceled or filled
        if (
          order.status === OrderStatus.CANCELED ||
          order.status === OrderStatus.FILLED ||
          order.status === OrderStatus.EXPIRED ||
          order.status === OrderStatus.REJECTED
        ) {
          throw new MinosSdkError(
            `Order is already ${order.status}`,
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the agent
        const agent = await this.getAgent(order.agentId);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${order.agentId.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can cancel orders',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Get the agent authority PDA
        const [agentAuthority, agentAuthorityBump] = await this.findAgentAuthorityAddress(
          new PublicKey(order.agentId)
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add cancel order instruction
        const cancelOrderIx = await this.buildCancelOrderInstruction(
          program,
          {
            orderId: publicKey,
            agentId: new PublicKey(order.agentId),
            agentAuthority,
            authority: this.client.provider.publicKey,
            agentAuthorityBump,
          }
        );
        tx.add(cancelOrderIx);
  
        // Send the transaction
        return await this.client.sendAndConfirmTransaction(tx, [], options);
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to cancel order: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Executes a trade using an AI agent
     * 
     * @param {ExecuteTradeParams} params - Trade execution parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TradeResult>} Trade result
     * @throws {MinosSdkError} If the trade execution fails
     * 
     * @example
     * ```typescript
     * const trade = await client.agent.executeTrade({
     *   agentId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   marketId: new PublicKey('9yKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsW'),
     *   side: OrderSide.BUY,
     *   quantity: new BN(1000000000),
     *   price: new BN(50000000)
     * });
     * console.log('Trade executed:', trade);
     * ```
     */
    public async executeTrade(
      params: ExecuteTradeParams,
      options: TransactionOptions = {}
    ): Promise<TradeResult> {
      try {
        // Validate parameters
        if (!params.agentId) {
          throw new MinosSdkError(
            'Agent ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.marketId) {
          throw new MinosSdkError(
            'Market ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.quantity || params.quantity.isZero()) {
          throw new MinosSdkError(
            'Trade quantity must be greater than zero',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the agent
        const agent = await this.getAgent(params.agentId);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${params.agentId.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the agent is active
        if (!agent.isActive) {
          throw new MinosSdkError(
            'Cannot execute trades with an inactive agent',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Check if the agent is linked to a vault
        if (!agent.vaultId) {
          throw new MinosSdkError(
            'Agent is not linked to any vault',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can execute trades',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Get the agent authority PDA
        const [agentAuthority, agentAuthorityBump] = await this.findAgentAuthorityAddress(
          new PublicKey(params.agentId)
        );
  
        // Generate a new keypair for the order/trade
        const tradeKeypair = Keypair.generate();
        const tradeId = tradeKeypair.publicKey;
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add execute trade instruction
        const executeTradeIx = await this.buildExecuteTradeInstruction(
          program,
          {
            tradeId,
            agentId: new PublicKey(params.agentId),
            agentAuthority,
            vaultId: new PublicKey(agent.vaultId),
            marketId: new PublicKey(params.marketId),
            authority: this.client.provider.publicKey,
            side: params.side,
            quantity: params.quantity,
            price: params.price || null,
            agentAuthorityBump,
          }
        );
        tx.add(executeTradeIx);
  
        // Send the transaction
        const signers = [tradeKeypair];
        const result = await this.client.sendAndConfirmTransaction(tx, signers, options);
  
        // Fetch the trade result
        const trade = await this.getTrade(tradeId);
        if (!trade) {
          throw new MinosSdkError(
            'Failed to fetch executed trade',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return trade;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to execute trade: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets a trade by its ID
     * 
     * @param {PublicKey | string} tradeId - Trade ID
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<TradeResult | null>} The trade result or null if not found
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const tradeId = new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU');
     * const trade = await client.agent.getTrade(tradeId);
     * if (trade) {
     *   console.log('Found trade:', trade);
     * } else {
     *   console.log('Trade not found');
     * }
     * ```
     */
    public async getTrade(
      tradeId: PublicKey | string,
      options: ApiRequestOptions = {}
    ): Promise<TradeResult | null> {
      try {
        const publicKey = validatePublicKey(tradeId);
  
        return await this.client.request<TradeResult>(
          'GET',
          `/v1/trades/${publicKey.toString()}`,
          undefined,
          options
        );
      } catch (error) {
        if (
          error instanceof MinosSdkError &&
          error.code === ErrorCode.NOT_FOUND
        ) {
          return null;
        }
  
        throw new MinosSdkError(
          `Failed to get trade: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all trades for an agent
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {TradeQueryParams} params - Query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<TradeResult>>} Paginated list of trades
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const trades = await client.agent.getTradesByAgent(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log(`This agent has ${trades.total} trades`);
     * ```
     */
    public async getTradesByAgent(
      agentId: PublicKey | string,
      params: TradeQueryParams = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<TradeResult>> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        // Convert PublicKey to string
        const queryParams: Record<string, any> = { ...params, agentId: publicKey.toString() };
  
        // Convert Date to timestamp if needed
        if (params.startDate) {
          if (params.startDate instanceof Date) {
            queryParams.startDate = Math.floor(params.startDate.getTime() / 1000);
          }
        }
        if (params.endDate) {
          if (params.endDate instanceof Date) {
            queryParams.endDate = Math.floor(params.endDate.getTime() / 1000);
          }
        }
  
        // Set default pagination
        if (!queryParams.page) {
          queryParams.page = 1;
        }
        if (!queryParams.limit) {
          queryParams.limit = 10;
        }
  
        return await this.client.request<PaginatedResponse<TradeResult>>(
          'GET',
          '/v1/trades',
          queryParams,
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get trades by agent: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Deletes an agent (only if it's not linked to any vault)
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TransactionResult>} Transaction result
     * @throws {MinosSdkError} If the deletion fails
     * 
     * @example
     * ```typescript
     * const result = await client.agent.deleteAgent(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log('Agent deleted:', result.signature);
     * ```
     */
    public async deleteAgent(
      agentId: PublicKey | string,
      options: TransactionOptions = {}
    ): Promise<TransactionResult> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        // Get the agent
        const agent = await this.getAgent(publicKey);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the agent is linked to a vault
        if (agent.vaultId) {
          throw new MinosSdkError(
            'Cannot delete an agent linked to a vault',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can delete the agent',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Find the strategy account
        const [strategyAccount] = await this.findStrategyAddress(publicKey);
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add delete agent instruction
        const deleteAgentIx = await this.buildDeleteAgentInstruction(
          program,
          {
            agentId: publicKey,
            authority: this.client.provider.publicKey,
            strategyAccount,
          }
        );
        tx.add(deleteAgentIx);
  
        // Send the transaction
        return await this.client.sendAndConfirmTransaction(tx, [], options);
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to delete agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    // ===== Utility methods =====
  
    /**
     * Finds the agent authority PDA address
     * 
     * @param {PublicKey} agentId - Agent ID
     * @returns {Promise<[PublicKey, number]>} Agent authority address and bump seed
     * @private
     */
    private async findAgentAuthorityAddress(
      agentId: PublicKey
    ): Promise<[PublicKey, number]> {
      return await PublicKey.findProgramAddress(
        [Buffer.from(AGENT_AUTHORITY_SEED), agentId.toBuffer()],
        this.client.programIds.aiAgentProgramId
      );
    }
  
    /**
     * Finds the strategy account PDA address
     * 
     * @param {PublicKey} agentId - Agent ID
     * @returns {Promise<[PublicKey, number]>} Strategy account address and bump seed
     * @private
     */
    private async findStrategyAddress(
      agentId: PublicKey
    ): Promise<[PublicKey, number]> {
      return await PublicKey.findProgramAddress(
        [Buffer.from(STRATEGY_SEED), agentId.toBuffer()],
        this.client.programIds.aiAgentProgramId
      );
    }
  
    /**
     * Finds the order account PDA address
     * 
     * @param {PublicKey} agentId - Agent ID
     * @param {PublicKey} orderId - Order ID
     * @returns {Promise<[PublicKey, number]>} Order account address and bump seed
     * @private
     */
    private async findOrderAddress(
      agentId: PublicKey,
      orderId: PublicKey
    ): Promise<[PublicKey, number]> {
      return await PublicKey.findProgramAddress(
        [Buffer.from(ORDER_SEED), agentId.toBuffer(), orderId.toBuffer()],
        this.client.programIds.aiAgentProgramId
      );
    }
  
    /**
     * Parses an agent account from the blockchain into the SDK format
     * 
     * @param {PublicKey} pubkey - Account public key
     * @param {any} agentAccount - Raw agent account data
     * @param {any} strategyAccount - Raw strategy account data
     * @returns {AgentAccount} Parsed agent account
     * @private
     */
    private parseAgentAccount(
      pubkey: PublicKey,
      agentAccount: any,
      strategyAccount: any
    ): AgentAccount {
      return {
        id: pubkey,
        name: agentAccount.name,
        description: agentAccount.description || '',
        authority: agentAccount.authority,
        modelType: this.parseAiModelType(agentAccount.modelType),
        strategyParams: deserializeStrategyParams(
          this.parseAiModelType(agentAccount.modelType),
          strategyAccount.strategyParams
        ),
        riskLevel: this.parseRiskLevel(agentAccount.riskLevel),
        timeHorizon: this.parseTimeHorizon(agentAccount.timeHorizon),
        vaultId: agentAccount.vaultId,
        performance: this.parseAgentPerformance(agentAccount.performance),
        createdAt: agentAccount.createdAt,
        updatedAt: agentAccount.updatedAt,
        isActive: agentAccount.isActive,
        lastExecutedAt: agentAccount.lastExecutedAt,
      };
    }
  
    /**
     * Parses an order account from the blockchain into the SDK format
     * 
     * @param {PublicKey} pubkey - Account public key
     * @param {any} account - Raw account data
     * @returns {OrderAccount} Parsed order account
     * @private
     */
    private parseOrderAccount(pubkey: PublicKey, account: any): OrderAccount {
      return {
        id: pubkey,
        vaultId: account.vaultId,
        agentId: account.agentId,
        marketId: account.marketId,
        orderType: this.parseOrderType(account.orderType),
        side: this.parseOrderSide(account.side),
        quantity: account.quantity,
        originalQuantity: account.originalQuantity,
        filledQuantity: account.filledQuantity,
        price: account.price,
        stopPrice: account.stopPrice,
        status: this.parseOrderStatus(account.status),
        createdAt: account.createdAt,
        updatedAt: account.updatedAt,
        clientOrderId: account.clientOrderId,
      };
    }
  
    /**
     * Parses the AI model type enum
     * 
     * @param {number} value - Raw AI model type value
     * @returns {AiModelType} Parsed AI model type
     * @private
     */
    private parseAiModelType(value: number): AiModelType {
      switch (value) {
        case 0:
          return AiModelType.ARIADNE;
        case 1:
          return AiModelType.ANDROGEUS;
        case 2:
          return AiModelType.DEUCALION;
        case 3:
          return AiModelType.CUSTOM;
        default:
          return AiModelType.CUSTOM;
      }
    }
  
    /**
     * Parses the risk level enum
     * 
     * @param {number} value - Raw risk level value
     * @returns {RiskLevel} Parsed risk level
     * @private
     */
    private parseRiskLevel(value: number): RiskLevel {
      switch (value) {
        case 0:
          return RiskLevel.CONSERVATIVE;
        case 1:
          return RiskLevel.MODERATE;
        case 2:
          return RiskLevel.AGGRESSIVE;
        case 3:
          return RiskLevel.CUSTOM;
        default:
          return RiskLevel.MODERATE;
      }
    }
  
    /**
     * Parses the time horizon enum
     * 
     * @param {number} value - Raw time horizon value
     * @returns {TimeHorizon} Parsed time horizon
     * @private
     */
    private parseTimeHorizon(value: number): TimeHorizon {
      switch (value) {
        case 0:
          return TimeHorizon.SHORT_TERM;
        case 1:
          return TimeHorizon.MEDIUM_TERM;
        case 2:
          return TimeHorizon.LONG_TERM;
        default:
          return TimeHorizon.MEDIUM_TERM;
      }
    }
  
    /**
     * Parses the order type enum
     * 
     * @param {number} value - Raw order type value
     * @returns {OrderType} Parsed order type
     * @private
     */
    private parseOrderType(value: number): OrderType {
      switch (value) {
        case 0:
          return OrderType.MARKET;
        case 1:
          return OrderType.LIMIT;
        case 2:
          return OrderType.STOP;
        case 3:
          return OrderType.STOP_LIMIT;
        case 4:
          return OrderType.TRAILING_STOP;
        default:
          return OrderType.MARKET;
      }
    }
  
    /**
     * Parses the order side enum
     * 
     * @param {number} value - Raw order side value
     * @returns {OrderSide} Parsed order side
     * @private
     */
    private parseOrderSide(value: number): OrderSide {
      switch (value) {
        case 0:
          return OrderSide.BUY;
        case 1:
          return OrderSide.SELL;
        default:
          return OrderSide.BUY;
      }
    }
  
    /**
     * Parses the order status enum
     * 
     * @param {number} value - Raw order status value
     * @returns {OrderStatus} Parsed order status
     * @private
     */
    private parseOrderStatus(value: number): OrderStatus {
      switch (value) {
        case 0:
          return OrderStatus.PENDING;
        case 1:
          return OrderStatus.OPEN;
        case 2:
          return OrderStatus.FILLED;
        case 3:
          return OrderStatus.PARTIALLY_FILLED;
        case 4:
          return OrderStatus.CANCELED;
        case 5:
          return OrderStatus.EXPIRED;
        case 6:
          return OrderStatus.REJECTED;
        default:
          return OrderStatus.PENDING;
      }
    }
  
    /**
     * Parses the agent performance data
     * 
     * @param {any} performance - Raw performance data
     * @returns {AgentPerformance} Parsed agent performance
     * @private
     */
    private parseAgentPerformance(performance: any): AgentPerformance {
      return {
        winRate: performance.winRate / 100,
        totalTrades: performance.totalTrades.toNumber(),
        profitFactor: performance.profitFactor / 100,
        averageReturn: performance.averageReturn / 100,
        maxConsecutiveWins: performance.maxConsecutiveWins.toNumber(),
        maxConsecutiveLosses: performance.maxConsecutiveLosses.toNumber(),
      };
    }
  
    // ===== Instruction builders =====
  
    /**
     * Builds an instruction to create an agent
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCreateAgentInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .createAgent(
          params.name,
          params.description,
          params.modelType,
          params.riskLevel,
          params.timeHorizon,
          params.vaultId,
          params.strategyParams
        )
        .accounts({
          agent: params.agentId,
          agentAuthority: params.agentAuthority,
          authority: params.authority,
          creator: params.creator,
          strategyAccount: params.strategyAccount,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to update an agent
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildUpdateAgentInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .updateAgent(
          params.name,
          params.description,
          params.isActive,
          params.vaultId,
          params.riskLevel,
          params.timeHorizon
        )
        .accounts({
          agent: params.agentId,
          authority: params.authority,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to update a strategy
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildUpdateStrategyInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .updateStrategy(params.strategyParams)
        .accounts({
          agent: params.agentId,
          authority: params.authority,
          strategyAccount: params.strategyAccount,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to create an order
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCreateOrderInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .createOrder(
          params.orderType,
          params.side,
          params.quantity,
          params.price,
          params.stopPrice,
          params.timeInForce,
          params.clientOrderId,
          params.agentAuthorityBump
        )
        .accounts({
          order: params.orderId,
          agent: params.agentId,
          agentAuthority: params.agentAuthority,
          vault: params.vaultId,
          market: params.marketId,
          authority: params.authority,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to cancel an order
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCancelOrderInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .cancelOrder(params.agentAuthorityBump)
        .accounts({
          order: params.orderId,
          agent: params.agentId,
          agentAuthority: params.agentAuthority,
          authority: params.authority,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to execute a trade
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildExecuteTradeInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .executeTrade(
          params.side,
          params.quantity,
          params.price,
          params.agentAuthorityBump
        )
        .accounts({
          trade: params.tradeId,
          agent: params.agentId,
          agentAuthority: params.agentAuthority,
          vault: params.vaultId,
          market: params.marketId,
          authority: params.authority,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to delete an agent
     * 
     * @param {Program} program - Agent program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildDeleteAgentInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .deleteAgent()
        .accounts({
          agent: params.agentId,
          authority: params.authority,
          strategyAccount: params.strategyAccount,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
    }
  }
  
  /**
   * Interface for agent update parameters
   */
  export interface UpdateAgentParams {
    /** New agent name */
    name?: string;
    /** New agent description */
    description?: string;
    /** Whether the agent is active */
    isActive?: boolean;
    /** Associated vault ID */
    vaultId?: PublicKey | null;
    /** Risk level */
    riskLevel?: RiskLevel;
    /** Time horizon */
    timeHorizon?: TimeHorizon;
  }
  
  /**
   * Interface for trade execution parameters
   */
  export interface ExecuteTradeParams {
    /** Agent ID */
    agentId: PublicKey;
    /** Market ID */
    marketId: PublicKey;
    /** Order side */
    side: OrderSide;
    /** Asset quantity */
    quantity: BN;
    /** Price (optional for market orders) */
    price?: BN;
  }
  
  /**
   * Interface for trade query parameters
   */
  export interface TradeQueryParams {
    /** Filter by agent ID */
    agentId?: PublicKey;
    /** Filter by vault ID */
    vaultId?: PublicKey;
    /** Filter by market ID */
    marketId?: PublicKey;
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
   * Interface for trade result
   */
  export interface TradeResult {
    /** Trade ID */
    id: PublicKey;
    /** Agent ID */
    agentId: PublicKey;
    /** Vault ID */
    vaultId: PublicKey;
    /** Market ID */
    marketId: PublicKey;
    /** Order side */
    side: OrderSide;
    /** Asset quantity */
    quantity: BN;
    /** Price */
    price: BN;
    /** Trading fee */
    fee: BN;
    /** Total value */
    totalValue: BN;
    /** Trade status */
    status: 'COMPLETED' | 'FAILED' | 'PENDING';
    /** Execution timestamp */
    executedAt: BN;
    /** Transaction signature */
    signature: string;
    /** Error message (if failed) */
    errorMessage?: string;
  }
  
  // Re-export the AgentClient class
  export { AgentClient };
  
  // Export utility functions
  export {
    validatePublicKey,
    serializeStrategyParams,
    deserializeStrategyParams,
  }; this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Find the strategy account
        const [strategyAccount] = await this.findStrategyAddress(publicKey);
  
        // Serialize strategy parameters
        const serializedStrategyParams = serializeStrategyParams(
          agent.modelType,
          strategyParams
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add update strategy instruction
        const updateStrategyIx = await this.buildUpdateStrategyInstruction(
          program,
          {
            agentId: publicKey,
            authority: this.client.provider.publicKey,
            strategyAccount,
            strategyParams: serializedStrategyParams,
          }
        );
        tx.add(updateStrategyIx);
  
        // Send the transaction
        await this.client.sendAndConfirmTransaction(tx, [], options);
  
        // Fetch the updated agent
        const updatedAgent = await this.getAgent(publicKey);
        if (!updatedAgent) {
          throw new MinosSdkError(
            'Failed to fetch updated agent',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return updatedAgent;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to update strategy: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Updates an agent's settings
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {Partial<UpdateAgentParams>} params - Update parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<AgentAccount>} Updated agent account
     * @throws {MinosSdkError} If the update fails
     * 
     * @example
     * ```typescript
     * const updatedAgent = await client.agent.updateAgent(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   {
     *     name: 'My Renamed Agent',
     *     description: 'Updated description',
     *     isActive: false
     *   }
     * );
     * console.log('Updated agent:', updatedAgent);
     * ```
     */
    public async updateAgent(
      agentId: PublicKey | string,
      params: Partial<UpdateAgentParams>,
      options: TransactionOptions = {}
    ): Promise<AgentAccount> {
      try {
        const publicKey = validatePublicKey(agentId);
  
        // Get the agent
        const agent = await this.getAgent(publicKey);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can update the agent',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.aiAgentProgramId
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add update agent instruction
        const updateAgentIx = await this.buildUpdateAgentInstruction(
          program,
          {
            agentId: publicKey,
            authority: this.client.provider.publicKey,
            name: params.name,
            description: params.description,
            isActive: params.isActive,
            vaultId: params.vaultId,
            riskLevel: params.riskLevel,
            timeHorizon: params.timeHorizon,
          }
        );
        tx.add(updateAgentIx);
  
        // Send the transaction
        await this.client.sendAndConfirmTransaction(tx, [], options);
  
        // Fetch the updated agent
        const updatedAgent = await this.getAgent(publicKey);
        if (!updatedAgent) {
          throw new MinosSdkError(
            'Failed to fetch updated agent',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return updatedAgent;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to update agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Activates or deactivates an agent
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {boolean} isActive - Whether the agent should be active
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<AgentAccount>} Updated agent account
     * @throws {MinosSdkError} If the activation/deactivation fails
     * 
     * @example
     * ```typescript
     * // Activate an agent
     * const activatedAgent = await client.agent.setAgentActive(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   true
     * );
     * console.log('Agent activated:', activatedAgent);
     * ```
     */
    public async setAgentActive(
      agentId: PublicKey | string,
      isActive: boolean,
      options: TransactionOptions = {}
    ): Promise<AgentAccount> {
      return await this.updateAgent(agentId, { isActive }, options);
    }
  
    /**
     * Links an agent to a vault
     * 
     * @param {PublicKey | string} agentId - Agent ID
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<AgentAccount>} Updated agent account
     * @throws {MinosSdkError} If the linking fails
     * 
     * @example
     * ```typescript
     * const linkedAgent = await client.agent.linkAgentToVault(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   new PublicKey('8yKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsV')
     * );
     * console.log('Agent linked to vault:', linkedAgent);
     * ```
     */
    public async linkAgentToVault(
      agentId: PublicKey | string,
      vaultId: PublicKey | string,
      options: TransactionOptions = {}
    ): Promise<AgentAccount> {
      const vaultPublicKey = validatePublicKey(vaultId);
      return await this.updateAgent(agentId, { vaultId: vaultPublicKey }, options);
    }
  
    /**
     * Creates a trade order using an agent
     * 
     * @param {CreateOrderParams} params - Order creation parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<OrderAccount>} The created order account
     * @throws {MinosSdkError} If order creation fails
     * 
     * @example
     * ```typescript
     * const order = await client.agent.createOrder({
     *   agentId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   vaultId: new PublicKey('8yKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsV'),
     *   marketId: new PublicKey('9yKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsW'),
     *   orderType: OrderType.LIMIT,
     *   side: OrderSide.BUY,
     *   quantity: new BN(1000000000),
     *   price: new BN(50000000),
     *   clientOrderId: 'my-order-1'
     * });
     * console.log('Created order:', order);
     * ```
     */
    public async createOrder(
      params: CreateOrderParams,
      options: TransactionOptions = {}
    ): Promise<OrderAccount> {
      try {
        // Validate parameters
        if (!params.agentId) {
          throw new MinosSdkError(
            'Agent ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.vaultId) {
          throw new MinosSdkError(
            'Vault ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.marketId) {
          throw new MinosSdkError(
            'Market ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.quantity || params.quantity.isZero()) {
          throw new MinosSdkError(
            'Order quantity must be greater than zero',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // For limit orders, price is required
        if (params.orderType === OrderType.LIMIT && (!params.price || params.price.isZero())) {
          throw new MinosSdkError(
            'Price is required for limit orders',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // For stop orders, stop price is required
        if (
          (params.orderType === OrderType.STOP || params.orderType === OrderType.STOP_LIMIT) &&
          (!params.stopPrice || params.stopPrice.isZero())
        ) {
          throw new MinosSdkError(
            'Stop price is required for stop orders',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the agent
        const agent = await this.getAgent(params.agentId);
        if (!agent) {
          throw new MinosSdkError(
            `Agent not found: ${params.agentId.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the agent is active
        if (!agent.isActive) {
          throw new MinosSdkError(
            'Cannot create orders with an inactive agent',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Check if the agent is linked to the specified vault
        if (!agent.vaultId || !agent.vaultId.equals(new PublicKey(params.vaultId))) {
          throw new MinosSdkError(
            'Agent is not linked to the specified vault',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(agent.authority))) {
          throw new MinosSdkError(
            'Only the agent authority can create orders',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await