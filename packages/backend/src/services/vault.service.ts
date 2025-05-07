/**
 * @fileoverview Vault Service for Minos-AI DeFi Platform
 * @description Handles interactions with Solana vaults through Anchor program interface
 * @author Minos-AI Engineering Team <engineering@minos-ai.io>
 * @copyright 2024 Minos-AI Labs, Inc.
 * @license MIT
 */

import { 
    Connection, 
    Keypair, 
    PublicKey, 
    Transaction, 
    SystemProgram, 
    sendAndConfirmTransaction, 
    TransactionSignature,
    ConfirmOptions,
    LAMPORTS_PER_SOL,
    Commitment
  } from '@solana/web3.js';
  import { 
    Program, 
    AnchorProvider, 
    Wallet, 
    BN, 
    IdlAccounts,
    web3 
  } from '@project-serum/anchor';
  import {
    TOKEN_PROGRAM_ID,
    ASSOCIATED_TOKEN_PROGRAM_ID,
    createAssociatedTokenAccountInstruction,
    getAssociatedTokenAddress,
    createInitializeMintInstruction,
    MINT_SIZE,
    getMinimumBalanceForRentExemptMint,
    createMintToInstruction,
    getMint,
    getAccount
  } from '@solana/spl-token';
  import * as bs58 from 'bs58';
  import { createLogger } from '../utils/logger';
  import { SolanaService } from '../utils/solana';
  import { config } from '../config';
  import NodeCache from 'node-cache';
  import { EventEmitter } from 'events';
  import { 
    StrategyType, 
    VaultStatus, 
    TokenMetadata, 
    VaultAccountWithMetadata, 
    VaultCreateParams, 
    VaultDepositParams, 
    VaultWithdrawParams, 
    StrategyExecutionParams,
    VaultStats,
    PerformanceMetrics,
    RiskMetrics
  } from '../interfaces/vault.interfaces';
  import { HttpError, SolanaTransactionError } from '../utils/errors';
  import { IDL, Vault } from '../types/vault';
  import { Redis } from 'ioredis';
  import { redisClient } from './cache.service';
  import { PerformanceService } from './performance.service';
  import { RiskManagementService } from './risk-management.service';
  import { UserWalletService } from './user-wallet.service';
  import { NotificationService } from './notification.service';
  import { QueueService } from './queue.service';
  
  // Initialize logger
  const logger = createLogger('vault-service');
  
  // Cache for vault data to reduce RPC calls
  const vaultCache = new NodeCache({
    stdTTL: 300, // 5 minutes
    checkperiod: 60, // Check every minute for expired items
    maxKeys: 1000
  });
  
  // Initialize event emitter for vault events
  const vaultEvents = new EventEmitter();
  
  /**
   * Service for interacting with Minos-AI vaults on Solana
   */
  export class VaultService {
    private program: Program<Vault>;
    private connection: Connection;
    private adminWallet: Keypair;
    private provider: AnchorProvider;
    private solanaService: SolanaService;
    private performanceService: PerformanceService;
    private riskService: RiskManagementService;
    private walletService: UserWalletService;
    private notificationService: NotificationService;
    private queueService: QueueService;
    private redis?: Redis;
    private confirmOptions: ConfirmOptions;
    
    /**
     * Initialize the Vault Service
     */
    constructor() {
      this.solanaService = new SolanaService();
      this.connection = this.solanaService.getConnection();
      
      // Initialize admin wallet from private key
      const walletSecretKey = bs58.decode(config.solana.walletPrivateKey);
      this.adminWallet = Keypair.fromSecretKey(walletSecretKey);
      
      // Create Anchor provider
      this.provider = new AnchorProvider(
        this.connection,
        new Wallet(this.adminWallet),
        {
          preflightCommitment: config.solana.commitment as Commitment,
          commitment: config.solana.commitment as Commitment
        }
      );
      
      // Initialize Anchor program
      this.program = new Program<Vault>(
        IDL,
        new PublicKey(config.vault.programId),
        this.provider
      );
      
      // Initialize confirm options
      this.confirmOptions = {
        skipPreflight: false,
        preflightCommitment: config.solana.commitment as Commitment,
        commitment: config.solana.commitment as Commitment
      };
      
      // Initialize supporting services
      this.performanceService = new PerformanceService();
      this.riskService = new RiskManagementService();
      this.walletService = new UserWalletService();
      this.notificationService = new NotificationService();
      this.queueService = new QueueService();
      
      // Initialize Redis if enabled
      if (config.redis.enabled && redisClient) {
        this.redis = redisClient;
      }
      
      // Set up event listeners
      this.setupEventListeners();
      
      logger.info(`VaultService initialized with program ID: ${config.vault.programId}`);
    }
    
    /**
     * Set up event listeners for vault events
     * @private
     */
    private setupEventListeners(): void {
      // Strategy execution events
      vaultEvents.on('strategy:executed', async (vaultPubkey: string, success: boolean, txId?: string) => {
        logger.info(`Strategy execution ${success ? 'succeeded' : 'failed'} for vault: ${vaultPubkey}`);
        
        if (success && txId) {
          // Update performance metrics
          await this.performanceService.updatePerformanceMetrics(vaultPubkey);
          
          // Send notification to vault depositors
          await this.notificationService.notifyStrategyExecution(vaultPubkey, txId);
          
          // Invalidate cache for this vault
          this.invalidateVaultCache(vaultPubkey);
        }
      });
      
      // Deposit events
      vaultEvents.on('deposit:processed', async (vaultPubkey: string, userPubkey: string, amount: number, txId: string) => {
        logger.info(`Deposit of ${amount} processed for user ${userPubkey} in vault ${vaultPubkey}`);
        
        // Update user position
        await this.updateUserPosition(vaultPubkey, userPubkey);
        
        // Invalidate cache
        this.invalidateVaultCache(vaultPubkey);
      });
      
      // Withdraw events
      vaultEvents.on('withdraw:processed', async (vaultPubkey: string, userPubkey: string, amount: number, txId: string) => {
        logger.info(`Withdrawal of ${amount} processed for user ${userPubkey} in vault ${vaultPubkey}`);
        
        // Update user position
        await this.updateUserPosition(vaultPubkey, userPubkey);
        
        // Invalidate cache
        this.invalidateVaultCache(vaultPubkey);
      });
      
      // Risk threshold events
      vaultEvents.on('risk:threshold-exceeded', async (vaultPubkey: string, metric: string, value: number, threshold: number) => {
        logger.warn(`Risk threshold exceeded for vault ${vaultPubkey}: ${metric} = ${value} (threshold: ${threshold})`);
        
        // Notify administrators
        await this.notificationService.notifyRiskThresholdExceeded(vaultPubkey, metric, value, threshold);
        
        // Take protective actions if configured
        if (config.featureFlags.advancedRiskMetrics) {
          await this.riskService.executeRiskManagementStrategy(vaultPubkey, metric, value);
        }
      });
    }
    
    /**
     * Create a new vault on Solana
     * @param {VaultCreateParams} params - Parameters for vault creation
     * @returns {Promise<VaultAccountWithMetadata>} - Created vault data
     */
    async createVault(params: VaultCreateParams): Promise<VaultAccountWithMetadata> {
      try {
        logger.info(`Creating new vault with name: ${params.name}`);
        
        // Generate a new keypair for the vault
        const vaultKeypair = Keypair.generate();
        const vaultPda = vaultKeypair.publicKey;
        
        // Generate token account for the vault
        const tokenMint = new PublicKey(params.tokenMint);
        const vaultTokenAccount = await getAssociatedTokenAddress(
          tokenMint,
          vaultPda,
          true // allowOwnerOffCurve
        );
        
        // Get token metadata
        const tokenMetadata = await this.solanaService.getTokenMetadata(tokenMint);
        
        // Get fees account for protocol fees
        const [feesAccount] = await PublicKey.findProgramAddressSync(
          [Buffer.from('fees')],
          this.program.programId
        );
        
        // Calculate the strategy type enum value
        let strategyType: number;
        switch (params.strategyType) {
          case StrategyType.YIELD_FARMING:
            strategyType = 0;
            break;
          case StrategyType.LIQUIDITY_PROVIDING:
            strategyType = 1;
            break;
          case StrategyType.LEVERAGE_TRADING:
            strategyType = 2;
            break;
          case StrategyType.ARBITRAGE:
            strategyType = 3;
            break;
          default:
            strategyType = 0;
        }
        
        // Build transaction for vault creation
        const tx = new Transaction();
        
        // Create associated token account for vault if it doesn't exist
        try {
          await getAccount(this.connection, vaultTokenAccount);
        } catch (error) {
          tx.add(
            createAssociatedTokenAccountInstruction(
              this.adminWallet.publicKey,
              vaultTokenAccount,
              vaultPda,
              tokenMint
            )
          );
        }
        
        // Add initialize vault instruction
        tx.add(
          await this.program.methods
            .initializeVault(
              params.name,
              new BN(params.managementFeePercentage * 100), // Convert to basis points (e.g., 2.5% -> 250)
              new BN(params.performanceFeePercentage * 100), // Convert to basis points
              new BN(strategyType),
              params.description || ""
            )
            .accounts({
              vault: vaultPda,
              tokenMint: tokenMint,
              vaultTokenAccount: vaultTokenAccount,
              feeAccount: feesAccount,
              authority: this.adminWallet.publicKey,
              systemProgram: SystemProgram.programId,
              tokenProgram: TOKEN_PROGRAM_ID,
            })
            .instruction()
        );
        
        // Sign and send transaction
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [this.adminWallet, vaultKeypair],
          this.confirmOptions
        );
        
        logger.info(`Vault created successfully. Signature: ${signature}`);
        
        // Fetch the created vault account
        const vaultAccount = await this.program.account.vault.fetch(vaultPda);
        
        // Format and return vault data with metadata
        const vaultData: VaultAccountWithMetadata = {
          pubkey: vaultPda.toString(),
          account: {
            name: vaultAccount.name,
            authority: vaultAccount.authority.toString(),
            tokenMint: vaultAccount.tokenMint.toString(),
            vaultTokenAccount: vaultAccount.vaultTokenAccount.toString(),
            totalDeposits: new BN(vaultAccount.totalDeposits).toNumber() / Math.pow(10, tokenMetadata.decimals),
            strategyType: strategyType as StrategyType,
            status: vaultAccount.status as VaultStatus,
            managementFeePercentage: vaultAccount.managementFeePercentage.toNumber() / 100,
            performanceFeePercentage: vaultAccount.performanceFeePercentage.toNumber() / 100,
            lastExecutedAt: vaultAccount.lastExecutedAt ? new Date(vaultAccount.lastExecutedAt.toNumber() * 1000) : null,
            createdAt: new Date(vaultAccount.createdAt.toNumber() * 1000),
            description: vaultAccount.description,
            initialNav: vaultAccount.initialNav ? new BN(vaultAccount.initialNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
            currentNav: vaultAccount.currentNav ? new BN(vaultAccount.currentNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
          },
          metadata: {
            tokenMetadata: tokenMetadata,
            transactionSignature: signature,
          }
        };
        
        // Cache the vault data
        this.cacheVaultData(vaultPda.toString(), vaultData);
        
        // Emit event
        vaultEvents.emit('vault:created', vaultPda.toString(), params.name);
        
        return vaultData;
        
      } catch (error: any) {
        logger.error({ err: error }, 'Failed to create vault');
        throw new SolanaTransactionError('Failed to create vault', error);
      }
    }
    
    /**
     * Get vault by public key
     * @param {string} pubkey - Vault public key
     * @returns {Promise<VaultAccountWithMetadata>} - Vault data with metadata
     */
    async getVault(pubkey: string): Promise<VaultAccountWithMetadata> {
      try {
        // Try to get from cache first
        const cachedVault = this.getVaultFromCache(pubkey);
        if (cachedVault) {
          return cachedVault;
        }
        
        logger.debug(`Fetching vault data for ${pubkey}`);
        
        // Get vault account data
        const vaultPubkey = new PublicKey(pubkey);
        const vaultAccount = await this.program.account.vault.fetch(vaultPubkey);
        
        // Get token metadata
        const tokenMint = vaultAccount.tokenMint;
        const tokenMetadata = await this.solanaService.getTokenMetadata(tokenMint);
        
        // Format vault data with metadata
        const vaultData: VaultAccountWithMetadata = {
          pubkey: pubkey,
          account: {
            name: vaultAccount.name,
            authority: vaultAccount.authority.toString(),
            tokenMint: vaultAccount.tokenMint.toString(),
            vaultTokenAccount: vaultAccount.vaultTokenAccount.toString(),
            totalDeposits: new BN(vaultAccount.totalDeposits).toNumber() / Math.pow(10, tokenMetadata.decimals),
            strategyType: vaultAccount.strategyType as StrategyType,
            status: vaultAccount.status as VaultStatus,
            managementFeePercentage: vaultAccount.managementFeePercentage.toNumber() / 100,
            performanceFeePercentage: vaultAccount.performanceFeePercentage.toNumber() / 100,
            lastExecutedAt: vaultAccount.lastExecutedAt ? new Date(vaultAccount.lastExecutedAt.toNumber() * 1000) : null,
            createdAt: new Date(vaultAccount.createdAt.toNumber() * 1000),
            description: vaultAccount.description,
            initialNav: vaultAccount.initialNav ? new BN(vaultAccount.initialNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
            currentNav: vaultAccount.currentNav ? new BN(vaultAccount.currentNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
          },
          metadata: {
            tokenMetadata: tokenMetadata,
          }
        };
        
        // Cache the vault data
        this.cacheVaultData(pubkey, vaultData);
        
        return vaultData;
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to fetch vault data for ${pubkey}`);
        throw new HttpError(404, 'Vault Not Found', `Vault with pubkey ${pubkey} not found or is invalid`);
      }
    }
    
    /**
     * Get all vaults
     * @param {boolean} includeInactive - Whether to include inactive vaults
     * @returns {Promise<VaultAccountWithMetadata[]>} - Array of vault data with metadata
     */
    async getAllVaults(includeInactive: boolean = false): Promise<VaultAccountWithMetadata[]> {
      try {
        logger.debug('Fetching all vaults');
        
        // Get all vault accounts from the program
        const vaultAccounts = await this.program.account.vault.all();
        
        // Process and format vault data with metadata
        const vaultDataPromises = vaultAccounts
          .filter(account => includeInactive || account.account.status !== VaultStatus.DECOMMISSIONED)
          .map(async account => {
            const pubkey = account.publicKey.toString();
            
            // Try to get from cache first
            const cachedVault = this.getVaultFromCache(pubkey);
            if (cachedVault) {
              return cachedVault;
            }
            
            // Get token metadata
            const tokenMint = account.account.tokenMint;
            const tokenMetadata = await this.solanaService.getTokenMetadata(tokenMint);
            
            // Format vault data with metadata
            const vaultData: VaultAccountWithMetadata = {
              pubkey: pubkey,
              account: {
                name: account.account.name,
                authority: account.account.authority.toString(),
                tokenMint: account.account.tokenMint.toString(),
                vaultTokenAccount: account.account.vaultTokenAccount.toString(),
                totalDeposits: new BN(account.account.totalDeposits).toNumber() / Math.pow(10, tokenMetadata.decimals),
                strategyType: account.account.strategyType as StrategyType,
                status: account.account.status as VaultStatus,
                managementFeePercentage: account.account.managementFeePercentage.toNumber() / 100,
                performanceFeePercentage: account.account.performanceFeePercentage.toNumber() / 100,
                lastExecutedAt: account.account.lastExecutedAt ? new Date(account.account.lastExecutedAt.toNumber() * 1000) : null,
                createdAt: new Date(account.account.createdAt.toNumber() * 1000),
                description: account.account.description,
                initialNav: account.account.initialNav ? new BN(account.account.initialNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
                currentNav: account.account.currentNav ? new BN(account.account.currentNav).toNumber() / Math.pow(10, tokenMetadata.decimals) : 1,
              },
              metadata: {
                tokenMetadata: tokenMetadata,
              }
            };
            
            // Cache the vault data
            this.cacheVaultData(pubkey, vaultData);
            
            return vaultData;
          });
        
        return await Promise.all(vaultDataPromises);
        
      } catch (error: any) {
        logger.error({ err: error }, 'Failed to fetch all vaults');
        throw new HttpError(500, 'Internal Server Error', 'Failed to fetch vault data');
      }
    }
    
    /**
     * Deposit tokens into a vault
     * @param {VaultDepositParams} params - Deposit parameters
     * @returns {Promise<TransactionSignature>} - Transaction signature
     */
    async deposit(params: VaultDepositParams): Promise<TransactionSignature> {
      try {
        logger.info(`Processing deposit of ${params.amount} to vault ${params.vaultPubkey} for user ${params.userPubkey}`);
        
        // Validate inputs
        if (params.amount <= 0) {
          throw new HttpError(400, 'Invalid Amount', 'Deposit amount must be greater than zero');
        }
        
        // Get vault data
        const vaultPubkey = new PublicKey(params.vaultPubkey);
        const vaultAccount = await this.program.account.vault.fetch(vaultPubkey);
        
        // Verify vault is active
        if (vaultAccount.status !== VaultStatus.ACTIVE) {
          throw new HttpError(400, 'Inactive Vault', 'Cannot deposit to an inactive vault');
        }
        
        // Get user's token account
        const userPubkey = new PublicKey(params.userPubkey);
        const tokenMint = vaultAccount.tokenMint;
        const userTokenAccount = await getAssociatedTokenAddress(
          tokenMint,
          userPubkey
        );
        
        // Get token metadata for decimal adjustment
        const tokenMetadata = await this.solanaService.getTokenMetadata(tokenMint);
        
        // Calculate amount in token units
        const amountInTokenUnits = new BN(
          params.amount * Math.pow(10, tokenMetadata.decimals)
        );
        
        // Get user's share token account
        const [vaultSharesMint] = await PublicKey.findProgramAddressSync(
          [vaultPubkey.toBuffer(), Buffer.from('shares')],
          this.program.programId
        );
        
        const userSharesAccount = await getAssociatedTokenAddress(
          vaultSharesMint,
          userPubkey
        );
        
        // Build transaction
        const tx = new Transaction();
        
        // Create associated token account for shares if it doesn't exist
        try {
          await getAccount(this.connection, userSharesAccount);
        } catch (error) {
          tx.add(
            createAssociatedTokenAccountInstruction(
              this.adminWallet.publicKey,
              userSharesAccount,
              userPubkey,
              vaultSharesMint
            )
          );
        }
        
        // Add deposit instruction
        tx.add(
          await this.program.methods
            .deposit(amountInTokenUnits)
            .accounts({
              vault: vaultPubkey,
              vaultTokenAccount: vaultAccount.vaultTokenAccount,
              userTokenAccount: userTokenAccount,
              vaultSharesMint: vaultSharesMint,
              userSharesAccount: userSharesAccount,
              user: userPubkey,
              tokenProgram: TOKEN_PROGRAM_ID,
            })
            .instruction()
        );
        
        // Sign and send transaction
        // Note: This would normally be signed by the user client-side
        // For testing/admin purposes, we're using the admin wallet
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [this.adminWallet], // In real world, the user would sign this
          this.confirmOptions
        );
        
        logger.info(`Deposit processed successfully. Signature: ${signature}`);
        
        // Emit event
        vaultEvents.emit('deposit:processed', params.vaultPubkey, params.userPubkey, params.amount, signature);
        
        // Invalidate cache for this vault
        this.invalidateVaultCache(params.vaultPubkey);
        
        return signature;
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to process deposit to vault ${params.vaultPubkey}`);
        throw new SolanaTransactionError('Failed to process deposit', error);
      }
    }
    
    /**
     * Withdraw tokens from a vault
     * @param {VaultWithdrawParams} params - Withdrawal parameters
     * @returns {Promise<TransactionSignature>} - Transaction signature
     */
    async withdraw(params: VaultWithdrawParams): Promise<TransactionSignature> {
      try {
        logger.info(`Processing withdrawal of ${params.amount} from vault ${params.vaultPubkey} for user ${params.userPubkey}`);
        
        // Validate inputs
        if (params.amount <= 0) {
          throw new HttpError(400, 'Invalid Amount', 'Withdrawal amount must be greater than zero');
        }
        
        // Get vault data
        const vaultPubkey = new PublicKey(params.vaultPubkey);
        const vaultAccount = await this.program.account.vault.fetch(vaultPubkey);
        
        // Get token metadata for decimal adjustment
        const tokenMint = vaultAccount.tokenMint;
        const tokenMetadata = await this.solanaService.getTokenMetadata(tokenMint);
        
        // Calculate amount in token units
        const amountInTokenUnits = new BN(
          params.amount * Math.pow(10, tokenMetadata.decimals)
        );
        
        // Get user accounts
        const userPubkey = new PublicKey(params.userPubkey);
        const userTokenAccount = await getAssociatedTokenAddress(
          tokenMint,
          userPubkey
        );
        
        // Get shares mint and user's share account
        const [vaultSharesMint] = await PublicKey.findProgramAddressSync(
          [vaultPubkey.toBuffer(), Buffer.from('shares')],
          this.program.programId
        );
        
        const userSharesAccount = await getAssociatedTokenAddress(
          vaultSharesMint,
          userPubkey
        );
        
        // Add withdraw instruction
        const tx = new Transaction();
        tx.add(
          await this.program.methods
            .withdraw(amountInTokenUnits)
            .accounts({
              vault: vaultPubkey,
              vaultTokenAccount: vaultAccount.vaultTokenAccount,
              userTokenAccount: userTokenAccount,
              vaultSharesMint: vaultSharesMint,
              userSharesAccount: userSharesAccount,
              user: userPubkey,
              tokenProgram: TOKEN_PROGRAM_ID,
            })
            .instruction()
        );
        
        // Sign and send transaction
        // Note: This would normally be signed by the user client-side
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [this.adminWallet], // In real world, the user would sign this
          this.confirmOptions
        );
        
        logger.info(`Withdrawal processed successfully. Signature: ${signature}`);
        
        // Emit event
        vaultEvents.emit('withdraw:processed', params.vaultPubkey, params.userPubkey, params.amount, signature);
        
        // Invalidate cache for this vault
        this.invalidateVaultCache(params.vaultPubkey);
        
        return signature;
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to process withdrawal from vault ${params.vaultPubkey}`);
        throw new SolanaTransactionError('Failed to process withdrawal', error);
      }
    }
    
    /**
     * Execute a vault's investment strategy
     * @param {StrategyExecutionParams} params - Strategy execution parameters
     * @returns {Promise<TransactionSignature>} - Transaction signature
     */
    async executeStrategy(params: StrategyExecutionParams): Promise<TransactionSignature> {
      try {
        logger.info(`Executing strategy for vault ${params.vaultPubkey}`);
        
        // Get vault data
        const vaultPubkey = new PublicKey(params.vaultPubkey);
        const vaultAccount = await this.program.account.vault.fetch(vaultPubkey);
        
        // Verify vault is active
        if (vaultAccount.status !== VaultStatus.ACTIVE) {
          throw new HttpError(400, 'Inactive Vault', 'Cannot execute strategy for an inactive vault');
        }
        
        // Check if strategy execution is already queued
        const isQueued = await this.queueService.isTaskQueued('strategy:execute', params.vaultPubkey);
        if (isQueued) {
          throw new HttpError(409, 'Strategy Execution Queued', 'Strategy execution is already queued for this vault');
        }
        
        // Get risk metrics before execution
        const riskMetrics = await this.riskService.getVaultRiskMetrics(params.vaultPubkey);
        
        // Check if any risk thresholds are exceeded
        const thresholdExceeded = this.riskService.checkRiskThresholds(riskMetrics);
        if (thresholdExceeded && !params.forceExecution) {
          throw new HttpError(400, 'Risk Threshold Exceeded', 'Cannot execute strategy due to risk threshold being exceeded');
        }
        
        // Build transaction based on strategy type
        const tx = new Transaction();
        
        switch (vaultAccount.strategyType as StrategyType) {
          case StrategyType.YIELD_FARMING:
            // Add yield farming strategy instructions
            tx.add(
              await this.program.methods
                .executeYieldFarmingStrategy(new BN(params.additionalParams?.targetApy || 0))
                .accounts({
                  vault: vaultPubkey,
                  vaultTokenAccount: vaultAccount.vaultTokenAccount,
                  authority: this.adminWallet.publicKey,
                  tokenProgram: TOKEN_PROGRAM_ID,
                })
                .instruction()
            );
            break;
            
          case StrategyType.LIQUIDITY_PROVIDING:
            // Add liquidity providing strategy instructions
            tx.add(
              await this.program.methods
                .executeLiquidityProvidingStrategy()
                .accounts({
                  vault: vaultPubkey,
                  vaultTokenAccount: vaultAccount.vaultTokenAccount,
                  authority: this.adminWallet.publicKey,
                  tokenProgram: TOKEN_PROGRAM_ID,
                })
                .instruction()
            );
            break;
            
          case StrategyType.LEVERAGE_TRADING:
            // Add leverage trading strategy instructions
            tx.add(
              await this.program.methods
                .executeLeverageTradingStrategy(new BN(params.additionalParams?.leverage || 0))
                .accounts({
                  vault: vaultPubkey,
                  vaultTokenAccount: vaultAccount.vaultTokenAccount,
                  authority: this.adminWallet.publicKey,
                  tokenProgram: TOKEN_PROGRAM_ID,
                })
                .instruction()
            );
            break;
            
          case StrategyType.ARBITRAGE:
            // Add arbitrage strategy instructions
            tx.add(
              await this.program.methods
                .executeArbitrageStrategy()
                .accounts({
                  vault: vaultPubkey,
                  vaultTokenAccount: vaultAccount.vaultTokenAccount,
                  authority: this.adminWallet.publicKey,
                  tokenProgram: TOKEN_PROGRAM_ID,
                })
                .instruction()
            );
            break;
            
          default:
            throw new HttpError(400, 'Invalid Strategy Type', 'Unknown strategy type for this vault');
        }
        
        // Sign and send transaction
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [this.adminWallet],
          this.confirmOptions
        );
        
        logger.info(`Strategy executed successfully. Signature: ${signature}`);
        
        // Emit event
        vaultEvents.emit('strategy:executed', params.vaultPubkey, true, signature);
        
        // Update vault stats after execution
        await this.updateVaultStats(params.vaultPubkey);
        
        // Update performance metrics
        await this.performanceService.updatePerformanceMetrics(params.vaultPubkey);
        
        // Invalidate cache for this vault
        this.invalidateVaultCache(params.vaultPubkey);
        
        return signature;
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to execute strategy for vault ${params.vaultPubkey}`);
        
        // Emit failure event
        vaultEvents.emit('strategy:executed', params.vaultPubkey, false);
        
        throw new SolanaTransactionError('Failed to execute strategy', error);
      }
    }
    
    /**
     * Get vault statistics and performance metrics
     * @param {string} vaultPubkey - Vault public key
     * @returns {Promise<VaultStats>} - Vault statistics
     */
    async getVaultStats(vaultPubkey: string): Promise<VaultStats> {
      try {
        // Try to get from cache first
        const cacheKey = `vault:stats:${vaultPubkey}`;
        const cachedStats = vaultCache.get<VaultStats>(cacheKey);
        if (cachedStats) {
          return cachedStats;
        }
        
        // Get vault data
        const vault = await this.getVault(vaultPubkey);
        
        // Get performance metrics
        const performanceMetrics = await this.performanceService.getPerformanceMetrics(vaultPubkey);
        
        // Get risk metrics
        const riskMetrics = await this.riskService.getVaultRiskMetrics(vaultPubkey);
        
        // Calculate additional statistics
        const depositorsCount = await this.getDepositorsCount(vaultPubkey);
        
        // Compile statistics
        const stats: VaultStats = {
          vaultPubkey,
          name: vault.account.name,
          tokenSymbol: vault.metadata.tokenMetadata.symbol,
          tvl: vault.account.totalDeposits,
          depositorsCount,
          performance: performanceMetrics,
          risk: riskMetrics,
          lastUpdated: new Date(),
        };
        
        // Cache the stats
        vaultCache.set(cacheKey, stats, 300); // Cache for 5 minutes
        
        return stats;
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get vault stats for ${vaultPubkey}`);
        throw new HttpError(500, 'Internal Server Error', 'Failed to retrieve vault statistics');
      }
    }
    
    /**
     * Update vault statistics (called after operations that change vault state)
     * @param {string} vaultPubkey - Vault public key
     * @returns {Promise<void>}
     * @private
     */
    private async updateVaultStats(vaultPubkey: string): Promise<void> {
      try {
        // Invalidate cache
        const cacheKey = `vault:stats:${vaultPubkey}`;
        vaultCache.del(cacheKey);
        
        // Get updated stats (which will be cached)
        await this.getVaultStats(vaultPubkey);
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to update vault stats for ${vaultPubkey}`);
        // Don't throw error as this is a background task
      }
    }
    
    /**
     * Get count of depositors for a vault
     * @param {string} vaultPubkey - Vault public key
     * @returns {Promise<number>} - Number of depositors
     * @private
     */
    private async getDepositorsCount(vaultPubkey: string): Promise<number> {
      try {
        // Get shares mint
        const vaultPubkeyObj = new PublicKey(vaultPubkey);
        const [vaultSharesMint] = await PublicKey.findProgramAddressSync(
          [vaultPubkeyObj.toBuffer(), Buffer.from('shares')],
          this.program.programId
        );
        
        // Get mint info
        const mintInfo = await getMint(this.connection, vaultSharesMint);
        
        // Return number of token holders (excluding mint authority)
        return Math.max(0, mintInfo.supply.gt(new BN(0)) ? (Number(mintInfo.supply) - 1) : 0);
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get depositors count for vault ${vaultPubkey}`);
        return 0;
      }
    }
    
    /**
     * Update user position after deposit/withdrawal
     * @param {string} vaultPubkey - Vault public key
     * @param {string} userPubkey - User public key
     * @returns {Promise<void>}
     * @private
     */
    private async updateUserPosition(vaultPubkey: string, userPubkey: string): Promise<void> {
      try {
        // Get shares mint
        const vaultPubkeyObj = new PublicKey(vaultPubkey);
        const [vaultSharesMint] = await PublicKey.findProgramAddressSync(
          [vaultPubkeyObj.toBuffer(), Buffer.from('shares')],
          this.program.programId
        );
        
        // Get user's share account
        const userPubkeyObj = new PublicKey(userPubkey);
        const userSharesAccount = await getAssociatedTokenAddress(
          vaultSharesMint,
          userPubkeyObj
        );
        
        // Get account info
        try {
          const accountInfo = await getAccount(this.connection, userSharesAccount);
          
          // Get vault data for NAV calculation
          const vault = await this.getVault(vaultPubkey);
          
          // Calculate user's share value
          const userShares = Number(accountInfo.amount);
          const userValue = userShares * vault.account.currentNav;
          
          // Store user position data (e.g., in database)
          // This would typically be done in a separate user position service
          
          logger.debug(`Updated user position for ${userPubkey} in vault ${vaultPubkey}: shares=${userShares}, value=${userValue}`);
          
        } catch (error) {
          // User might not have any shares
          logger.debug(`No shares found for user ${userPubkey} in vault ${vaultPubkey}`);
        }
        
      } catch (error: any) {
        logger.error({ err: error }, `Failed to update user position for ${userPubkey} in vault ${vaultPubkey}`);
        // Don't throw error as this is a background task
      }
    }
    
    /**
     * Cache vault data for faster retrieval
     * @param {string} pubkey - Vault public key
     * @param {VaultAccountWithMetadata} data - Vault data
     * @private
     */
    private cacheVaultData(pubkey: string, data: VaultAccountWithMetadata): void {
      const cacheKey = `vault:${pubkey}`;
      vaultCache.set(cacheKey, data);
    }
    
    /**
     * Get vault data from cache
     * @param {string} pubkey - Vault public key
     * @returns {VaultAccountWithMetadata | undefined} - Cached vault data or undefined if not in cache
     * @private
     */
    private getVaultFromCache(pubkey: string): VaultAccountWithMetadata | undefined {
      const cacheKey = `vault:${pubkey}`;
      return vaultCache.get<VaultAccountWithMetadata>(cacheKey);
    }
    
    /**
     * Invalidate vault cache
     * @param {string} pubkey - Vault public key
     * @private
     */
    private invalidateVaultCache(pubkey: string): void {
      const vaultCacheKey = `vault:${pubkey}`;
      const statsCacheKey = `vault:stats:${pubkey}`;
      vaultCache.del(vaultCacheKey);
      vaultCache.del(statsCacheKey);
    }
    
    /**
     * Schedule strategy execution for all active vaults
     * Used by the background task scheduler
     * @returns {Promise<void>}
     */
    async scheduleAllVaultStrategyExecutions(): Promise<void> {
      try {
        logger.info('Scheduling strategy executions for all active vaults');
        
        // Get all active vaults
        const vaults = await this.getAllVaults(false);
        const activeVaults = vaults.filter(v => v.account.status === VaultStatus.ACTIVE);
        
        logger.info(`Found ${activeVaults.length} active vaults for strategy execution`);
        
        // Queue strategy execution for each active vault
        for (const vault of activeVaults) {
          // Check if enough time has passed since last execution
          const lastExecuted = vault.account.lastExecutedAt;
          const now = new Date();
          const minInterval = config.vault.strategyExecutionInterval;
          
          if (!lastExecuted || (now.getTime() - lastExecuted.getTime() >= minInterval)) {
            await this.queueService.queueTask('strategy:execute', vault.pubkey, {
              vaultPubkey: vault.pubkey,
              forceExecution: false
            });
            
            logger.debug(`Queued strategy execution for vault ${vault.pubkey} (${vault.account.name})`);
          } else {
            logger.debug(`Skipping strategy execution for vault ${vault.pubkey} (${vault.account.name}) - executed recently`);
          }
        }
        
      } catch (error: any) {
        logger.error({ err: error }, 'Failed to schedule strategy executions');
        // Don't throw error as this is a background task
      }
    }
  }
  
  // Export singleton instance
  export const vaultService = new VaultService();
  
  // Export event emitter for external subscribers
  export { vaultEvents };