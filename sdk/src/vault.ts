/**
 * @file Vault client implementation
 * @module @minos-ai/sdk/vault
 * @description Provides functionality for interacting with Minos-AI vaults
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
  import { 
    Token, 
    TOKEN_PROGRAM_ID, 
    ASSOCIATED_TOKEN_PROGRAM_ID, 
    AccountLayout as TokenAccountLayout,
  } from '@solana/spl-token';
  import { Buffer } from 'buffer';
  
  // Import client base and types
  import { MinosClient } from './client';
  import {
    VaultAccount,
    CreateVaultParams,
    DepositParams,
    WithdrawParams,
    VaultQueryParams,
    VaultStatus,
    AssetType,
    PaginatedResponse,
    TransactionResult,
    TransactionOptions,
    ApiRequestOptions,
    ErrorCode,
    MinosSdkError,
    VaultPerformance,
    AccountWithData,
    ApiResponse,
    AiModelType,
    FeeStructure,
  } from './types';
  
  // Import constants and utilities
  import {
    DEFAULT_COMMITMENT,
    VAULT_SEED,
    VAULT_AUTHORITY_SEED,
    VAULT_TOKEN_ACCOUNT_SEED,
    INVESTOR_SEED,
    DEFAULT_FEE_STRUCTURE,
  } from './constants';
  import { 
    getAssociatedTokenAddress, 
    createAssociatedTokenAccountInstruction,
    verifyAmount,
    validatePublicKey,
  } from './utils';
  
  /**
   * VaultClient provides methods for interacting with vaults in the Minos-AI platform
   */
  export class VaultClient {
    private readonly client: MinosClient;
  
    /**
     * Creates a new VaultClient instance
     * 
     * @param {MinosClient} client - MinosClient instance
     */
    constructor(client: MinosClient) {
      this.client = client;
    }
  
    /**
     * Creates a new vault
     * 
     * @param {CreateVaultParams} params - Vault creation parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<VaultAccount>} The created vault account
     * @throws {MinosSdkError} If vault creation fails
     * 
     * @example
     * ```typescript
     * const vault = await client.vault.createVault({
     *   name: 'My AI Trading Vault',
     *   initialDeposit: new BN(1000000000), // 1 SOL in lamports
     *   assetType: AssetType.SOL,
     *   riskLevel: RiskLevel.MODERATE,
     *   timeHorizon: TimeHorizon.MEDIUM_TERM,
     *   aiModelType: AiModelType.ARIADNE
     * });
     * console.log('Created vault:', vault);
     * ```
     */
    public async createVault(
      params: CreateVaultParams,
      options: TransactionOptions = {}
    ): Promise<VaultAccount> {
      try {
        // Validate parameters
        if (!params.name) {
          throw new MinosSdkError(
            'Vault name is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.initialDeposit || params.initialDeposit.isZero()) {
          throw new MinosSdkError(
            'Initial deposit amount must be greater than zero',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the vault program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Generate a new keypair for the vault if not provided
        const vaultKeypair = Keypair.generate();
        const vaultId = vaultKeypair.publicKey;
  
        // Derive the vault authority PDA
        const [vaultAuthority, vaultAuthorityBump] = await this.findVaultAuthorityAddress(
          vaultId
        );
  
        // Derive the vault token account PDA for SPL tokens
        let vaultTokenAccount: PublicKey | null = null;
        if (params.assetType === AssetType.SPL_TOKEN) {
          if (!params.tokenMint) {
            throw new MinosSdkError(
              'Token mint must be provided for SPL token vaults',
              ErrorCode.VALIDATION_ERROR
            );
          }
  
          vaultTokenAccount = await this.findVaultTokenAccount(
            vaultId,
            new PublicKey(params.tokenMint)
          );
        }
  
        // Set default fee structure if not provided
        const fees = params.fees || DEFAULT_FEE_STRUCTURE;
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add create vault instruction
        const createVaultIx = await this.buildCreateVaultInstruction(
          program,
          {
            vaultId,
            vaultAuthority,
            creator: this.client.provider.publicKey,
            tokenMint: params.tokenMint,
            vaultTokenAccount,
            name: params.name,
            assetType: params.assetType,
            riskLevel: params.riskLevel,
            timeHorizon: params.timeHorizon,
            aiModelType: params.aiModelType,
            initialDeposit: params.initialDeposit,
            fees,
            maxCapacity: params.maxCapacity,
            minDeposit: params.minDeposit,
            allowWithdrawals: params.allowWithdrawals !== false,
            isPublic: params.isPublic !== false,
            strategyParams: params.strategyParams || {},
          }
        );
        tx.add(createVaultIx);
  
        // If this is an SPL token vault, create and fund the token account
        if (params.assetType === AssetType.SPL_TOKEN && params.tokenMint && vaultTokenAccount) {
          // Find the user's associated token account
          const userTokenAccount = await getAssociatedTokenAddress(
            new PublicKey(params.tokenMint),
            this.client.provider.publicKey
          );
  
          // Check if the user's token account exists
          const userTokenAccountExists = await this.client.accountExists(userTokenAccount);
          if (!userTokenAccountExists) {
            // Create the user's token account if it doesn't exist
            const createUserTokenAccountIx = createAssociatedTokenAccountInstruction(
              this.client.provider.publicKey,
              userTokenAccount,
              this.client.provider.publicKey,
              new PublicKey(params.tokenMint)
            );
            tx.add(createUserTokenAccountIx);
          }
  
          // Create vault token account
          const createVaultTokenAccountIx = await this.buildCreateTokenAccountInstruction(
            vaultTokenAccount,
            new PublicKey(params.tokenMint),
            vaultAuthority
          );
          tx.add(createVaultTokenAccountIx);
  
          // Transfer tokens to the vault
          const transferTokensIx = await this.buildTokenTransferInstruction(
            userTokenAccount,
            vaultTokenAccount,
            this.client.provider.publicKey,
            params.initialDeposit
          );
          tx.add(transferTokensIx);
        }
  
        // Send the transaction
        const signers = [vaultKeypair];
        const result = await this.client.sendAndConfirmTransaction(tx, signers, options);
  
        // Fetch the created vault
        const vault = await this.getVault(vaultId);
        if (!vault) {
          throw new MinosSdkError(
            'Failed to fetch created vault',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return vault;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to create vault: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets a vault by its ID
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<VaultAccount | null>} The vault account or null if not found
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const vaultId = new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU');
     * const vault = await client.vault.getVault(vaultId);
     * if (vault) {
     *   console.log('Found vault:', vault);
     * } else {
     *   console.log('Vault not found');
     * }
     * ```
     */
    public async getVault(
      vaultId: PublicKey | string,
      options: ApiRequestOptions = {}
    ): Promise<VaultAccount | null> {
      try {
        const publicKey = validatePublicKey(vaultId);
  
        // Attempt to fetch from the blockchain
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        try {
          const vaultAccount = await program.account.vault.fetch(publicKey);
          return this.parseVaultAccount(publicKey, vaultAccount);
        } catch (error) {
          // If not found on-chain, try API
          try {
            return await this.client.request<VaultAccount>(
              'GET',
              `/v1/vaults/${publicKey.toString()}`,
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
          `Failed to get vault: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all vaults matching the specified query parameters
     * 
     * @param {VaultQueryParams} params - Query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<VaultAccount>>} Paginated list of vaults
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const vaults = await client.vault.getVaults({
     *   status: VaultStatus.ACTIVE,
     *   assetType: AssetType.SOL,
     *   sortBy: 'aum',
     *   sortDirection: 'desc',
     *   page: 1,
     *   limit: 10
     * });
     * console.log(`Found ${vaults.total} vaults, showing ${vaults.items.length}`);
     * ```
     */
    public async getVaults(
      params: VaultQueryParams = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<VaultAccount>> {
      try {
        // Convert PublicKey to string
        const queryParams: Record<string, any> = { ...params };
        if (params.authority && params.authority instanceof PublicKey) {
          queryParams.authority = params.authority.toString();
        }
  
        // Set default pagination
        if (!queryParams.page) {
          queryParams.page = 1;
        }
        if (!queryParams.limit) {
          queryParams.limit = 10;
        }
  
        return await this.client.request<PaginatedResponse<VaultAccount>>(
          'GET',
          '/v1/vaults',
          queryParams,
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vaults: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all vaults owned by the specified authority
     * 
     * @param {PublicKey | string} authority - Vault authority
     * @param {VaultQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<VaultAccount>>} Paginated list of vaults
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const myVaults = await client.vault.getVaultsByAuthority(client.provider.publicKey);
     * console.log(`You own ${myVaults.total} vaults`);
     * ```
     */
    public async getVaultsByAuthority(
      authority: PublicKey | string,
      params: Omit<VaultQueryParams, 'authority'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<VaultAccount>> {
      try {
        const publicKey = validatePublicKey(authority);
        return await this.getVaults({ ...params, authority: publicKey }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vaults by authority: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all vaults using a specific AI model type
     * 
     * @param {AiModelType} aiModelType - AI model type
     * @param {VaultQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<VaultAccount>>} Paginated list of vaults
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const ariadneVaults = await client.vault.getVaultsByModelType(AiModelType.ARIADNE);
     * console.log(`Found ${ariadneVaults.total} vaults using Ariadne model`);
     * ```
     */
    public async getVaultsByModelType(
      aiModelType: AiModelType,
      params: Omit<VaultQueryParams, 'aiModelType'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<VaultAccount>> {
      try {
        return await this.getVaults({ ...params, aiModelType }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vaults by model type: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all vaults with minimum performance
     * 
     * @param {number} minPerformance - Minimum performance (percentage)
     * @param {VaultQueryParams} params - Additional query parameters
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<VaultAccount>>} Paginated list of vaults
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * // Get vaults with at least 10% performance
     * const highPerformingVaults = await client.vault.getVaultsByMinPerformance(10);
     * console.log(`Found ${highPerformingVaults.total} high-performing vaults`);
     * ```
     */
    public async getVaultsByMinPerformance(
      minPerformance: number,
      params: Omit<VaultQueryParams, 'minPerformance'> = {},
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<VaultAccount>> {
      try {
        return await this.getVaults({ ...params, minPerformance }, options);
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vaults by min performance: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Deposits assets into a vault
     * 
     * @param {DepositParams} params - Deposit parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TransactionResult>} Transaction result
     * @throws {MinosSdkError} If the deposit fails
     * 
     * @example
     * ```typescript
     * // Deposit 1 SOL into a vault
     * const deposit = await client.vault.deposit({
     *   vaultId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   amount: new BN(1000000000) // 1 SOL in lamports
     * });
     * console.log('Deposit successful:', deposit.signature);
     * ```
     */
    public async deposit(
      params: DepositParams,
      options: TransactionOptions = {}
    ): Promise<TransactionResult> {
      try {
        // Validate parameters
        if (!params.vaultId) {
          throw new MinosSdkError(
            'Vault ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.amount || params.amount.isZero()) {
          throw new MinosSdkError(
            'Deposit amount must be greater than zero',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the vault
        const vault = await this.getVault(params.vaultId);
        if (!vault) {
          throw new MinosSdkError(
            `Vault not found: ${params.vaultId.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Get the vault authority PDA
        const [vaultAuthority] = await this.findVaultAuthorityAddress(
          new PublicKey(vault.id)
        );
  
        // Find the investor account PDA
        const [investorAccount, investorBump] = await this.findInvestorAddress(
          new PublicKey(vault.id),
          this.client.provider.publicKey
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Check if the investor account already exists
        const investorAccountExists = await this.client.accountExists(investorAccount);
        if (!investorAccountExists) {
          // Create investor account instruction
          const createInvestorIx = await this.buildCreateInvestorInstruction(
            program,
            {
              vaultId: new PublicKey(vault.id),
              investor: this.client.provider.publicKey,
              investorAccount,
            }
          );
          tx.add(createInvestorIx);
        }
  
        // Add deposit instruction based on asset type
        if (vault.assetType === AssetType.SOL) {
          // SOL deposit
          const depositIx = await this.buildSolDepositInstruction(
            program,
            {
              vaultId: new PublicKey(vault.id),
              vaultAuthority,
              investor: this.client.provider.publicKey,
              investorAccount,
              amount: params.amount,
            }
          );
          tx.add(depositIx);
        } else if (vault.assetType === AssetType.SPL_TOKEN && vault.tokenMint) {
          // SPL token deposit
          const tokenMint = new PublicKey(vault.tokenMint);
          const vaultTokenAccount = await this.findVaultTokenAccount(
            new PublicKey(vault.id),
            tokenMint
          );
  
          // Find the user's token account
          const fromTokenAccount = params.fromTokenAccount ||
            await getAssociatedTokenAddress(tokenMint, this.client.provider.publicKey);
  
          // Check if the user's token account exists
          const fromTokenAccountExists = await this.client.accountExists(fromTokenAccount);
          if (!fromTokenAccountExists) {
            throw new MinosSdkError(
              'Source token account does not exist',
              ErrorCode.ACCOUNT_NOT_FOUND
            );
          }
  
          // Add the token deposit instruction
          const depositIx = await this.buildTokenDepositInstruction(
            program,
            {
              vaultId: new PublicKey(vault.id),
              vaultAuthority,
              vaultTokenAccount,
              investor: this.client.provider.publicKey,
              investorAccount,
              fromTokenAccount,
              tokenMint,
              amount: params.amount,
            }
          );
          tx.add(depositIx);
        } else {
          throw new MinosSdkError(
            'Unsupported asset type',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Send the transaction
        return await this.client.sendAndConfirmTransaction(tx, [], options);
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to deposit: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Withdraws assets from a vault
     * 
     * @param {WithdrawParams} params - Withdrawal parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TransactionResult>} Transaction result
     * @throws {MinosSdkError} If the withdrawal fails
     * 
     * @example
     * ```typescript
     * // Withdraw 0.5 SOL from a vault
     * const withdrawal = await client.vault.withdraw({
     *   vaultId: new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   amount: new BN(500000000) // 0.5 SOL in lamports
     * });
     * console.log('Withdrawal successful:', withdrawal.signature);
     * ```
     */
    public async withdraw(
      params: WithdrawParams,
      options: TransactionOptions = {}
    ): Promise<TransactionResult> {
      try {
        // Validate parameters
        if (!params.vaultId) {
          throw new MinosSdkError(
            'Vault ID is required',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        if (!params.amount || params.amount.isZero()) {
          throw new MinosSdkError(
            'Withdrawal amount must be greater than zero',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the vault
        const vault = await this.getVault(params.vaultId);
        if (!vault) {
          throw new MinosSdkError(
            `Vault not found: ${params.vaultId.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if withdrawals are allowed
        if (vault.status !== VaultStatus.ACTIVE) {
          throw new MinosSdkError(
            `Cannot withdraw from vault with status: ${vault.status}`,
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Get the vault authority PDA
        const [vaultAuthority, vaultAuthorityBump] = await this.findVaultAuthorityAddress(
          new PublicKey(vault.id)
        );
  
        // Find the investor account PDA
        const [investorAccount, investorBump] = await this.findInvestorAddress(
          new PublicKey(vault.id),
          this.client.provider.publicKey
        );
  
        // Check if investor account exists
        const investorAccountExists = await this.client.accountExists(investorAccount);
        if (!investorAccountExists) {
          throw new MinosSdkError(
            'You are not an investor in this vault',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add withdrawal instruction based on asset type
        if (vault.assetType === AssetType.SOL) {
          // SOL withdrawal
          const withdrawIx = await this.buildSolWithdrawInstruction(
            program,
            {
              vaultId: new PublicKey(vault.id),
              vaultAuthority,
              investor: this.client.provider.publicKey,
              investorAccount,
              amount: params.amount,
              vaultAuthorityBump,
            }
          );
          tx.add(withdrawIx);
        } else if (vault.assetType === AssetType.SPL_TOKEN && vault.tokenMint) {
          // SPL token withdrawal
          const tokenMint = new PublicKey(vault.tokenMint);
          const vaultTokenAccount = await this.findVaultTokenAccount(
            new PublicKey(vault.id),
            tokenMint
          );
  
          // Find the user's token account
          const toTokenAccount = params.toTokenAccount ||
            await getAssociatedTokenAddress(tokenMint, this.client.provider.publicKey);
  
          // Check if the user's token account exists
          const toTokenAccountExists = await this.client.accountExists(toTokenAccount);
          if (!toTokenAccountExists) {
            // Create the user's token account if it doesn't exist
            const createToTokenAccountIx = createAssociatedTokenAccountInstruction(
              this.client.provider.publicKey,
              toTokenAccount,
              this.client.provider.publicKey,
              tokenMint
            );
            tx.add(createToTokenAccountIx);
          }
  
          // Add the token withdrawal instruction
          const withdrawIx = await this.buildTokenWithdrawInstruction(
            program,
            {
              vaultId: new PublicKey(vault.id),
              vaultAuthority,
              vaultTokenAccount,
              investor: this.client.provider.publicKey,
              investorAccount,
              toTokenAccount,
              tokenMint,
              amount: params.amount,
              vaultAuthorityBump,
            }
          );
          tx.add(withdrawIx);
        } else {
          throw new MinosSdkError(
            'Unsupported asset type',
            ErrorCode.VALIDATION_ERROR
          );
        }
  
        // Send the transaction
        return await this.client.sendAndConfirmTransaction(tx, [], options);
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to withdraw: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Updates a vault's settings
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {Partial<UpdateVaultParams>} params - Update parameters
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<VaultAccount>} Updated vault account
     * @throws {MinosSdkError} If the update fails
     * 
     * @example
     * ```typescript
     * const updatedVault = await client.vault.updateVault(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   {
     *     name: 'My Renamed Vault',
     *     allowWithdrawals: false
     *   }
     * );
     * console.log('Updated vault:', updatedVault);
     * ```
     */
    public async updateVault(
      vaultId: PublicKey | string,
      params: Partial<UpdateVaultParams>,
      options: TransactionOptions = {}
    ): Promise<VaultAccount> {
      try {
        const publicKey = validatePublicKey(vaultId);
  
        // Get the vault
        const vault = await this.getVault(publicKey);
        if (!vault) {
          throw new MinosSdkError(
            `Vault not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(vault.authority))) {
          throw new MinosSdkError(
            'Only the vault authority can update the vault',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add update vault instruction
        const updateVaultIx = await this.buildUpdateVaultInstruction(
          program,
          {
            vaultId: publicKey,
            authority: this.client.provider.publicKey,
            ...params,
          }
        );
        tx.add(updateVaultIx);
  
        // Send the transaction
        await this.client.sendAndConfirmTransaction(tx, [], options);
  
        // Fetch the updated vault
        const updatedVault = await this.getVault(publicKey);
        if (!updatedVault) {
          throw new MinosSdkError(
            'Failed to fetch updated vault',
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        return updatedVault;
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to update vault: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Closes a vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {TransactionOptions} options - Transaction options
     * @returns {Promise<TransactionResult>} Transaction result
     * @throws {MinosSdkError} If the vault closure fails
     * 
     * @example
     * ```typescript
     * const result = await client.vault.closeVault(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log('Vault closed:', result.signature);
     * ```
     */
    public async closeVault(
      vaultId: PublicKey | string,
      options: TransactionOptions = {}
    ): Promise<TransactionResult> {
      try {
        const publicKey = validatePublicKey(vaultId);
  
        // Get the vault
        const vault = await this.getVault(publicKey);
        if (!vault) {
          throw new MinosSdkError(
            `Vault not found: ${publicKey.toString()}`,
            ErrorCode.ACCOUNT_NOT_FOUND
          );
        }
  
        // Check if the caller is the authority
        if (!this.client.provider.publicKey.equals(new PublicKey(vault.authority))) {
          throw new MinosSdkError(
            'Only the vault authority can close the vault',
            ErrorCode.UNAUTHORIZED
          );
        }
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Get the vault authority PDA
        const [vaultAuthority, vaultAuthorityBump] = await this.findVaultAuthorityAddress(
          publicKey
        );
  
        // Build the transaction
        const tx = new Transaction();
  
        // Add close vault instruction
        const closeVaultIx = await this.buildCloseVaultInstruction(
          program,
          {
            vaultId: publicKey,
            authority: this.client.provider.publicKey,
            vaultAuthority,
            vaultAuthorityBump,
          }
        );
        tx.add(closeVaultIx);
  
        // Send the transaction
        return await this.client.sendAndConfirmTransaction(tx, [], options);
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to close vault: ${error instanceof Error ? error.message : 'Unknown error'}`,
          ErrorCode.TRANSACTION_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets the performance history of a vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {string} period - Time period ('1d', '1w', '1m', '3m', '1y', 'all')
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<VaultPerformance>} Vault performance data
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const performance = await client.vault.getVaultPerformance(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   '1m'
     * );
     * console.log('Vault performance:', performance);
     * ```
     */
    public async getVaultPerformance(
      vaultId: PublicKey | string,
      period: '1d' | '1w' | '1m' | '3m' | '1y' | 'all' = '1m',
      options: ApiRequestOptions = {}
    ): Promise<VaultPerformance> {
      try {
        const publicKey = validatePublicKey(vaultId);
  
        return await this.client.request<VaultPerformance>(
          'GET',
          `/v1/vaults/${publicKey.toString()}/performance`,
          { period },
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vault performance: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets the investor account data for a vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {PublicKey | string} investor - Investor public key
     * @returns {Promise<InvestorAccount | null>} Investor account data or null if not found
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const investorAccount = await client.vault.getInvestorAccount(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'),
     *   client.provider.publicKey
     * );
     * if (investorAccount) {
     *   console.log('Your investment:', investorAccount);
     * } else {
     *   console.log('You are not an investor in this vault');
     * }
     * ```
     */
    public async getInvestorAccount(
      vaultId: PublicKey | string,
      investor: PublicKey | string
    ): Promise<InvestorAccount | null> {
      try {
        const vaultPublicKey = validatePublicKey(vaultId);
        const investorPublicKey = validatePublicKey(investor);
  
        // Find the investor account PDA
        const [investorAccount] = await this.findInvestorAddress(
          vaultPublicKey,
          investorPublicKey
        );
  
        // Get the program
        const program = await this.client.getProgram(
          this.client.programIds.vaultProgramId
        );
  
        // Fetch the investor account
        try {
          const account = await program.account.investor.fetch(investorAccount);
          return this.parseInvestorAccount(investorAccount, account);
        } catch (error) {
          // Account not found
          return null;
        }
      } catch (error) {
        if (error instanceof MinosSdkError) {
          throw error;
        }
  
        throw new MinosSdkError(
          `Failed to get investor account: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    /**
     * Gets all investors for a vault
     * 
     * @param {PublicKey | string} vaultId - Vault ID
     * @param {ApiRequestOptions} options - API request options
     * @returns {Promise<PaginatedResponse<InvestorAccount>>} Paginated list of investors
     * @throws {MinosSdkError} If the request fails
     * 
     * @example
     * ```typescript
     * const investors = await client.vault.getVaultInvestors(
     *   new PublicKey('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU')
     * );
     * console.log(`This vault has ${investors.total} investors`);
     * ```
     */
    public async getVaultInvestors(
      vaultId: PublicKey | string,
      options: ApiRequestOptions = {}
    ): Promise<PaginatedResponse<InvestorAccount>> {
      try {
        const publicKey = validatePublicKey(vaultId);
  
        return await this.client.request<PaginatedResponse<InvestorAccount>>(
          'GET',
          `/v1/vaults/${publicKey.toString()}/investors`,
          undefined,
          options
        );
      } catch (error) {
        throw new MinosSdkError(
          `Failed to get vault investors: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
          ErrorCode.API_ERROR,
          error instanceof Error ? error : undefined
        );
      }
    }
  
    // ===== Utility methods =====
  
    /**
     * Finds the vault authority PDA address
     * 
     * @param {PublicKey} vaultId - Vault ID
     * @returns {Promise<[PublicKey, number]>} Vault authority address and bump seed
     * @private
     */
    private async findVaultAuthorityAddress(
      vaultId: PublicKey
    ): Promise<[PublicKey, number]> {
      return await PublicKey.findProgramAddress(
        [Buffer.from(VAULT_AUTHORITY_SEED), vaultId.toBuffer()],
        this.client.programIds.vaultProgramId
      );
    }
  
    /**
     * Finds the vault token account PDA address
     * 
     * @param {PublicKey} vaultId - Vault ID
     * @param {PublicKey} tokenMint - Token mint
     * @returns {Promise<PublicKey>} Vault token account address
     * @private
     */
    private async findVaultTokenAccount(
      vaultId: PublicKey,
      tokenMint: PublicKey
    ): Promise<PublicKey> {
      const [address] = await PublicKey.findProgramAddress(
        [
          Buffer.from(VAULT_TOKEN_ACCOUNT_SEED),
          vaultId.toBuffer(),
          tokenMint.toBuffer(),
        ],
        this.client.programIds.vaultProgramId
      );
      return address;
    }
  
    /**
     * Finds the investor account PDA address
     * 
     * @param {PublicKey} vaultId - Vault ID
     * @param {PublicKey} investor - Investor public key
     * @returns {Promise<[PublicKey, number]>} Investor account address and bump seed
     * @private
     */
    private async findInvestorAddress(
      vaultId: PublicKey,
      investor: PublicKey
    ): Promise<[PublicKey, number]> {
      return await PublicKey.findProgramAddress(
        [Buffer.from(INVESTOR_SEED), vaultId.toBuffer(), investor.toBuffer()],
        this.client.programIds.vaultProgramId
      );
    }
  
    /**
     * Parses a vault account from the blockchain into the SDK format
     * 
     * @param {PublicKey} pubkey - Account public key
     * @param {any} account - Raw account data
     * @returns {VaultAccount} Parsed vault account
     * @private
     */
    private parseVaultAccount(pubkey: PublicKey, account: any): VaultAccount {
      return {
        id: pubkey,
        name: account.name,
        authority: account.authority,
        assetType: this.parseAssetType(account.assetType),
        tokenMint: account.tokenMint,
        tokenAccount: account.tokenAccount,
        status: this.parseVaultStatus(account.status),
        aum: account.aum,
        investorCount: account.investorCount.toNumber(),
        createdAt: account.createdAt,
        updatedAt: account.updatedAt,
        performance: this.parseVaultPerformance(account.performance),
        agentId: account.agentId,
        riskLevel: this.parseRiskLevel(account.riskLevel),
        fees: this.parseFeeStructure(account.fees),
      };
    }
  
    /**
     * Parses an investor account from the blockchain into the SDK format
     * 
     * @param {PublicKey} pubkey - Account public key
     * @param {any} account - Raw account data
     * @returns {InvestorAccount} Parsed investor account
     * @private
     */
    private parseInvestorAccount(pubkey: PublicKey, account: any): InvestorAccount {
      return {
        id: pubkey,
        vault: account.vault,
        investor: account.investor,
        amount: account.amount,
        shares: account.shares,
        depositedAt: account.depositedAt,
        lastWithdrawalAt: account.lastWithdrawalAt,
        totalDeposited: account.totalDeposited,
        totalWithdrawn: account.totalWithdrawn,
        profit: account.profit,
        profitPercentage: account.profitPercentage,
      };
    }
  
    /**
     * Parses the asset type enum
     * 
     * @param {number} value - Raw asset type value
     * @returns {AssetType} Parsed asset type
     * @private
     */
    private parseAssetType(value: number): AssetType {
      switch (value) {
        case 0:
          return AssetType.SOL;
        case 1:
          return AssetType.SPL_TOKEN;
        case 2:
          return AssetType.USDC;
        case 3:
          return AssetType.BTC;
        case 4:
          return AssetType.ETH;
        default:
          return AssetType.SOL;
      }
    }
  
    /**
     * Parses the vault status enum
     * 
     * @param {number} value - Raw vault status value
     * @returns {VaultStatus} Parsed vault status
     * @private
     */
    private parseVaultStatus(value: number): VaultStatus {
      switch (value) {
        case 0:
          return VaultStatus.INITIALIZED;
        case 1:
          return VaultStatus.ACTIVE;
        case 2:
          return VaultStatus.PAUSED;
        case 3:
          return VaultStatus.CLOSED;
        default:
          return VaultStatus.INITIALIZED;
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
     * Parses the fee structure
     * 
     * @param {any} fees - Raw fee structure
     * @returns {FeeStructure} Parsed fee structure
     * @private
     */
    private parseFeeStructure(fees: any): FeeStructure {
      return {
        managementFee: fees.managementFee / 100,
        performanceFee: fees.performanceFee / 100,
        withdrawalFee: fees.withdrawalFee / 100,
      };
    }
  
    /**
     * Parses the vault performance data
     * 
     * @param {any} performance - Raw performance data
     * @returns {VaultPerformance} Parsed vault performance
     * @private
     */
    private parseVaultPerformance(performance: any): VaultPerformance {
      return {
        totalReturn: performance.totalReturn / 100,
        annualizedReturn: performance.annualizedReturn / 100,
        sharpeRatio: performance.sharpeRatio / 100,
        maxDrawdown: performance.maxDrawdown / 100,
        dailyReturns: Array.isArray(performance.dailyReturns)
          ? performance.dailyReturns.map((v: number) => v / 100)
          : [],
        monthlyReturns: Array.isArray(performance.monthlyReturns)
          ? performance.monthlyReturns.map((v: number) => v / 100)
          : [],
      };
    }
  
    // ===== Instruction builders =====
  
    /**
     * Builds an instruction to create a vault
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCreateVaultInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .createVault(
          params.name,
          params.assetType,
          params.riskLevel,
          params.timeHorizon,
          params.aiModelType,
          params.initialDeposit,
          {
            managementFee: Math.floor(params.fees.managementFee * 100),
            performanceFee: Math.floor(params.fees.performanceFee * 100),
            withdrawalFee: Math.floor(params.fees.withdrawalFee * 100),
          },
          params.maxCapacity || new BN(0),
          params.minDeposit || new BN(0),
          params.allowWithdrawals !== false,
          params.isPublic !== false,
          params.strategyParams || {}
        )
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          authority: params.creator,
          tokenMint: params.tokenMint || null,
          vaultTokenAccount: params.vaultTokenAccount || null,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to create an investor account
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCreateInvestorInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .createInvestor()
        .accounts({
          vault: params.vaultId,
          investor: params.investor,
          investorAccount: params.investorAccount,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to deposit SOL
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildSolDepositInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .depositSol(params.amount)
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          investor: params.investor,
          investorAccount: params.investorAccount,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to deposit SPL tokens
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildTokenDepositInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .depositToken(params.amount)
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          vaultTokenAccount: params.vaultTokenAccount,
          investor: params.investor,
          investorAccount: params.investorAccount,
          fromTokenAccount: params.fromTokenAccount,
          tokenMint: params.tokenMint,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to withdraw SOL
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildSolWithdrawInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .withdrawSol(params.amount, params.vaultAuthorityBump)
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          investor: params.investor,
          investorAccount: params.investorAccount,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to withdraw SPL tokens
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildTokenWithdrawInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .withdrawToken(params.amount, params.vaultAuthorityBump)
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          vaultTokenAccount: params.vaultTokenAccount,
          investor: params.investor,
          investorAccount: params.investorAccount,
          toTokenAccount: params.toTokenAccount,
          tokenMint: params.tokenMint,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to update a vault
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildUpdateVaultInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .updateVault(
          params.name,
          params.status,
          {
            managementFee: params.fees ? Math.floor(params.fees.managementFee * 100) : null,
            performanceFee: params.fees ? Math.floor(params.fees.performanceFee * 100) : null,
            withdrawalFee: params.fees ? Math.floor(params.fees.withdrawalFee * 100) : null,
          },
          params.maxCapacity,
          params.minDeposit,
          params.allowWithdrawals,
          params.isPublic
        )
        .accounts({
          vault: params.vaultId,
          authority: params.authority,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to close a vault
     * 
     * @param {Program} program - Vault program
     * @param {Object} params - Instruction parameters
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCloseVaultInstruction(
      program: Program,
      params: any
    ): Promise<TransactionInstruction> {
      return await program.methods
        .closeVault(params.vaultAuthorityBump)
        .accounts({
          vault: params.vaultId,
          vaultAuthority: params.vaultAuthority,
          authority: params.authority,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
    }
  
    /**
     * Builds an instruction to create a token account
     * 
     * @param {PublicKey} tokenAccount - Token account address
     * @param {PublicKey} mint - Token mint
     * @param {PublicKey} owner - Token account owner
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildCreateTokenAccountInstruction(
      tokenAccount: PublicKey,
      mint: PublicKey,
      owner: PublicKey
    ): Promise<TransactionInstruction> {
      // Create associated token account instruction
      return createAssociatedTokenAccountInstruction(
        this.client.provider.publicKey,
        tokenAccount,
        owner,
        mint
      );
    }
  
    /**
     * Builds an instruction to transfer tokens
     * 
     * @param {PublicKey} source - Source token account
     * @param {PublicKey} destination - Destination token account
     * @param {PublicKey} owner - Token account owner
     * @param {BN} amount - Amount to transfer
     * @returns {Promise<TransactionInstruction>} Transaction instruction
     * @private
     */
    private async buildTokenTransferInstruction(
      source: PublicKey,
      destination: PublicKey,
      owner: PublicKey,
      amount: BN
    ): Promise<TransactionInstruction> {
      return Token.createTransferInstruction(
        TOKEN_PROGRAM_ID,
        source,
        destination,
        owner,
        [],
        amount.toNumber()
      );
    }
  }
  
  /**
   * Interface for vault update parameters
   */
  export interface UpdateVaultParams {
    /** New vault name */
    name?: string;
    /** New vault status */
    status?: VaultStatus;
    /** New fee structure */
    fees?: FeeStructure;
    /** New maximum capacity */
    maxCapacity?: BN;
    /** New minimum deposit */
    minDeposit?: BN;
    /** Whether to allow withdrawals */
    allowWithdrawals?: boolean;
    /** Whether the vault is public */
    isPublic?: boolean;
  }
  
  /**
   * Interface for investor account data
   */
  export interface InvestorAccount {
    /** Account ID */
    id: PublicKey;
    /** Vault ID */
    vault: PublicKey;
    /** Investor public key */
    investor: PublicKey;
    /** Current amount invested */
    amount: BN;
    /** Number of shares */
    shares: BN;
    /** Initial deposit timestamp */
    depositedAt: BN;
    /** Last withdrawal timestamp */
    lastWithdrawalAt: BN | null;
    /** Total amount deposited */
    totalDeposited: BN;
    /** Total amount withdrawn */
    totalWithdrawn: BN;
    /** Current profit */
    profit: BN;
    /** Profit percentage */
    profitPercentage: number;
  }
  
  // Re-export the VaultClient class
  export { VaultClient };
  
  // Export utility functions
  export { 
    getAssociatedTokenAddress, 
    createAssociatedTokenAccountInstruction,
    verifyAmount,
    validatePublicKey,
  };