/**
 * @fileoverview Solana Blockchain Utilities for Minos-AI
 * @description Core utilities for interacting with the Solana blockchain
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
    Commitment,
    GetVersionedTransactionConfig,
    VersionedTransactionResponse,
    TransactionError,
    ParsedTransactionWithMeta,
    ParsedInstruction,
    CompiledInnerInstruction,
    ComputeBudgetProgram,
    TransactionMessage,
    VersionedTransaction,
    TransactionVersion,
    Blockhash,
    BlockheightBasedTransactionConfirmationStrategy,
    GetProgramAccountsFilter,
    TokenBalance,
    AddressLookupTableAccount,
    AccountInfo,
    RpcResponseAndContext,
    SignatureStatus,
    SimulatedTransactionResponse,
    GetMultipleAccountsConfig,
    GetAccountInfoConfig,
    GetParsedProgramAccountsConfig,
    GetTransactionConfig,
    ParsedTransactionMeta,
    Context,
    TokenAmount,
    BlockResponse,
    GetBlockConfig,
    Finality
  } from '@solana/web3.js';
  
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
    getAccount,
    createTransferInstruction,
    getOrCreateAssociatedTokenAccount,
    TOKEN_2022_PROGRAM_ID,
    TokenAccountNotFoundError,
    transferChecked,
    createCloseAccountInstruction,
    createBurnCheckedInstruction,
    createApproveInstruction,
    createRevokeInstruction,
    TokenInvalidAccountOwnerError,
    TokenInvalidMintError,
    getAssociatedTokenAddressSync,
    createSyncNativeInstruction,
    createCloseAccountCheckedInstruction,
    mintTo,
    mintToChecked,
    transferCheckedWithFee,
    createWrappedNativeAccount
  } from '@solana/spl-token';
  
  import axios from 'axios';
  import * as bs58 from 'bs58';
  import { BN } from 'bn.js';
  import * as nacl from 'tweetnacl';
  import { createLogger } from './logger';
  import { config } from '../config';
  import { TokenMetadata } from '../interfaces/vault.interfaces';
  import { SolanaTransactionError, HttpError } from './errors';
  import { sleep } from './common';
  import { METAPLEX_METADATA_PROGRAM_ID } from '../constants';
  
  // Initialize logger
  const logger = createLogger('solana-utils');
  
  /**
   * Service for interacting with the Solana blockchain
   */
  export class SolanaService {
    private connection: Connection;
    private adminWallet: Keypair | null = null;
    private confirmOptions: ConfirmOptions;
    private connectionMap: Map<string, Connection> = new Map();
    private lookupTableCache: Map<string, AddressLookupTableAccount> = new Map();
    private recentBlockhashes: Map<string, { blockhash: string, lastValidBlockHeight: number }> = new Map();
    
    /**
     * Initialize the Solana Service
     * @param endpoint - Optional RPC endpoint to override config
     * @param commitment - Optional commitment level to override config
     */
    constructor(endpoint?: string, commitment?: Commitment) {
      const rpcEndpoint = endpoint || config.solana.rpcEndpoint;
      const commitmentLevel = commitment || (config.solana.commitment as Commitment);
      
      // Set default confirm options
      this.confirmOptions = {
        skipPreflight: false,
        preflightCommitment: commitmentLevel,
        commitment: commitmentLevel,
        maxRetries: 3
      };
      
      // Create connection with custom configurations
      this.connection = new Connection(rpcEndpoint, {
        commitment: commitmentLevel,
        confirmTransactionInitialTimeout: config.solana.connectionTimeoutMs,
        disableRetryOnRateLimit: false,
        httpHeaders: {
          'Content-Type': 'application/json',
          'User-Agent': 'Minos-AI/1.0.0'
        }
      });
      
      // Cache the connection
      this.connectionMap.set(`${rpcEndpoint}-${commitmentLevel}`, this.connection);
      
      // Initialize admin wallet if private key is provided in config
      if (config.solana.walletPrivateKey) {
        try {
          const walletSecretKey = bs58.decode(config.solana.walletPrivateKey);
          this.adminWallet = Keypair.fromSecretKey(walletSecretKey);
          logger.info(`Admin wallet initialized with public key: ${this.adminWallet.publicKey.toString()}`);
        } catch (error) {
          logger.error({ err: error }, 'Failed to initialize admin wallet from private key');
        }
      }
      
      logger.info(`SolanaService initialized with endpoint: ${rpcEndpoint} and commitment: ${commitmentLevel}`);
      
      // Schedule regular blockhash updates for transaction efficiency
      this.scheduleBlockhashUpdates();
    }
    
    /**
     * Get the Solana connection
     * @returns Connection instance
     */
    getConnection(): Connection {
      return this.connection;
    }
    
    /**
     * Get a connection with specific parameters
     * @param endpoint - RPC endpoint
     * @param commitment - Commitment level
     * @returns Connection instance
     */
    getConnectionWithParams(endpoint: string, commitment: Commitment): Connection {
      const key = `${endpoint}-${commitment}`;
      
      // Return cached connection if available
      if (this.connectionMap.has(key)) {
        return this.connectionMap.get(key)!;
      }
      
      // Create new connection
      const connection = new Connection(endpoint, {
        commitment: commitment,
        confirmTransactionInitialTimeout: config.solana.connectionTimeoutMs,
        disableRetryOnRateLimit: false
      });
      
      // Cache the connection
      this.connectionMap.set(key, connection);
      
      return connection;
    }
    
    /**
     * Get the admin wallet
     * @returns Admin wallet keypair or null if not initialized
     */
    getAdminWallet(): Keypair | null {
      return this.adminWallet;
    }
    
    /**
     * Create a new wallet (keypair)
     * @returns Created keypair and base58 encoded secret key
     */
    createWallet(): { keypair: Keypair, secretKey: string } {
      const keypair = Keypair.generate();
      const secretKey = bs58.encode(keypair.secretKey);
      
      return { keypair, secretKey };
    }
    
    /**
     * Airdrop SOL to a wallet (for devnet/testnet only)
     * @param recipient - Recipient wallet address
     * @param amount - Amount in SOL (default: 1)
     * @returns Transaction signature
     */
    async airdropSol(recipient: string, amount: number = 1): Promise<string> {
      if (config.solana.network === 'mainnet-beta') {
        throw new HttpError(400, 'Invalid Operation', 'Airdrop is not supported on mainnet');
      }
      
      try {
        const recipientPubkey = new PublicKey(recipient);
        const amountLamports = amount * LAMPORTS_PER_SOL;
        
        const signature = await this.connection.requestAirdrop(recipientPubkey, amountLamports);
        await this.connection.confirmTransaction(signature, config.solana.commitment as Commitment);
        
        logger.info(`Airdropped ${amount} SOL to ${recipient}. Signature: ${signature}`);
        
        return signature;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to airdrop SOL to ${recipient}`);
        throw new SolanaTransactionError(`Failed to airdrop SOL: ${error.message}`, error);
      }
    }
    
    /**
     * Get the balance of a wallet in SOL
     * @param wallet - Wallet address
     * @returns Balance in SOL
     */
    async getBalance(wallet: string): Promise<number> {
      try {
        const pubkey = new PublicKey(wallet);
        const balance = await this.connection.getBalance(pubkey);
        
        return balance / LAMPORTS_PER_SOL;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get balance for ${wallet}`);
        throw new HttpError(500, 'Solana Error', `Failed to get balance: ${error.message}`);
      }
    }
    
    /**
     * Get the token balance of a wallet
     * @param wallet - Wallet address
     * @param tokenMint - Token mint address
     * @returns Token balance
     */
    async getTokenBalance(wallet: string, tokenMint: string): Promise<number> {
      try {
        const walletPubkey = new PublicKey(wallet);
        const mintPubkey = new PublicKey(tokenMint);
        
        // Get associated token account
        const tokenAccount = await getAssociatedTokenAddress(
          mintPubkey,
          walletPubkey,
          false,
          TOKEN_PROGRAM_ID
        );
        
        try {
          // Get account info
          const accountInfo = await getAccount(
            this.connection,
            tokenAccount,
            config.solana.commitment as Commitment,
            TOKEN_PROGRAM_ID
          );
          
          // Get token metadata for decimals
          const tokenMetadata = await this.getTokenMetadata(mintPubkey);
          
          // Calculate balance with decimals
          const balance = Number(accountInfo.amount) / Math.pow(10, tokenMetadata.decimals);
          
          return balance;
        } catch (error: any) {
          if (error instanceof TokenAccountNotFoundError) {
            // Account doesn't exist, so balance is 0
            return 0;
          }
          throw error;
        }
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get token balance for ${wallet} (token: ${tokenMint})`);
        throw new HttpError(500, 'Solana Error', `Failed to get token balance: ${error.message}`);
      }
    }
    
    /**
     * Transfer SOL from one wallet to another
     * @param fromWallet - Sender wallet keypair
     * @param toAddress - Recipient wallet address
     * @param amount - Amount in SOL
     * @returns Transaction signature
     */
    async transferSol(fromWallet: Keypair, toAddress: string, amount: number): Promise<string> {
      try {
        const toPubkey = new PublicKey(toAddress);
        const amountLamports = Math.floor(amount * LAMPORTS_PER_SOL);
        
        // Create transaction
        const tx = new Transaction().add(
          SystemProgram.transfer({
            fromPubkey: fromWallet.publicKey,
            toPubkey,
            lamports: amountLamports
          })
        );
        
        // Get recent blockhash
        const { blockhash, lastValidBlockHeight } = await this.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = fromWallet.publicKey;
        
        // Sign and send transaction
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [fromWallet],
          this.confirmOptions
        );
        
        logger.info(`Transferred ${amount} SOL from ${fromWallet.publicKey.toString()} to ${toAddress}. Signature: ${signature}`);
        
        return signature;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to transfer SOL from ${fromWallet.publicKey.toString()} to ${toAddress}`);
        throw new SolanaTransactionError(`Failed to transfer SOL: ${error.message}`, error);
      }
    }
    
    /**
     * Transfer SPL tokens from one wallet to another
     * @param fromWallet - Sender wallet keypair
     * @param toAddress - Recipient wallet address
     * @param tokenMint - Token mint address
     * @param amount - Amount of tokens (in token units, not including decimals)
     * @returns Transaction signature
     */
    async transferToken(fromWallet: Keypair, toAddress: string, tokenMint: string, amount: number): Promise<string> {
      try {
        const toPubkey = new PublicKey(toAddress);
        const mintPubkey = new PublicKey(tokenMint);
        
        // Get sender token account
        const fromTokenAccount = await getAssociatedTokenAddress(
          mintPubkey,
          fromWallet.publicKey,
          false,
          TOKEN_PROGRAM_ID
        );
        
        // Get recipient token account
        const toTokenAccount = await getAssociatedTokenAddress(
          mintPubkey,
          toPubkey,
          false,
          TOKEN_PROGRAM_ID
        );
        
        // Create transaction
        const tx = new Transaction();
        
        // Check if recipient token account exists
        try {
          await getAccount(this.connection, toTokenAccount);
        } catch (error: any) {
          if (error instanceof TokenAccountNotFoundError) {
            // Create associated token account if it doesn't exist
            tx.add(
              createAssociatedTokenAccountInstruction(
                fromWallet.publicKey, // Payer
                toTokenAccount, // Associated token account
                toPubkey, // Owner
                mintPubkey, // Mint
                TOKEN_PROGRAM_ID,
                ASSOCIATED_TOKEN_PROGRAM_ID
              )
            );
          } else {
            throw error;
          }
        }
        
        // Get token metadata for decimals
        const tokenMetadata = await this.getTokenMetadata(mintPubkey);
        
        // Calculate amount with decimals
        const amountWithDecimals = Math.floor(amount * Math.pow(10, tokenMetadata.decimals));
        
        // Add transfer instruction
        tx.add(
          createTransferInstruction(
            fromTokenAccount, // Source
            toTokenAccount, // Destination
            fromWallet.publicKey, // Owner
            amountWithDecimals, // Amount
            [],
            TOKEN_PROGRAM_ID
          )
        );
        
        // Get recent blockhash
        const { blockhash, lastValidBlockHeight } = await this.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = fromWallet.publicKey;
        
        // Sign and send transaction
        const signature = await sendAndConfirmTransaction(
          this.connection,
          tx,
          [fromWallet],
          this.confirmOptions
        );
        
        logger.info(`Transferred ${amount} ${tokenMetadata.symbol} tokens from ${fromWallet.publicKey.toString()} to ${toAddress}. Signature: ${signature}`);
        
        return signature;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to transfer tokens from ${fromWallet.publicKey.toString()} to ${toAddress}`);
        throw new SolanaTransactionError(`Failed to transfer tokens: ${error.message}`, error);
      }
    }
    
    /**
     * Get metadata for a token
     * @param tokenMint - Token mint address or PublicKey
     * @returns Token metadata
     */
    async getTokenMetadata(tokenMint: string | PublicKey): Promise<TokenMetadata> {
      const mintPubkey = typeof tokenMint === 'string' ? new PublicKey(tokenMint) : tokenMint;
      
      try {
        // Get mint info
        const mintInfo = await getMint(
          this.connection,
          mintPubkey,
          config.solana.commitment as Commitment,
          TOKEN_PROGRAM_ID
        );
        
        // Get token metadata PDA
        const [metadataPDA] = await PublicKey.findProgramAddressSync(
          [
            Buffer.from('metadata'),
            METAPLEX_METADATA_PROGRAM_ID.toBuffer(),
            mintPubkey.toBuffer(),
          ],
          METAPLEX_METADATA_PROGRAM_ID
        );
        
        // Default metadata
        let metadata: TokenMetadata = {
          name: 'Unknown Token',
          symbol: 'UNKNOWN',
          description: '',
          image: '',
          decimals: mintInfo.decimals,
          supply: Number(mintInfo.supply) / Math.pow(10, mintInfo.decimals),
        };
        
        try {
          // Try to fetch metadata from Metaplex
          const metadataAccount = await this.connection.getAccountInfo(metadataPDA);
          
          if (metadataAccount) {
            // Parse metadata
            // Note: This is a simplified parser. In a production environment,
            // you would use a proper Metaplex SDK to parse this data.
            const data = metadataAccount.data;
            
            // Skip the first 1 + 32 + 32 + 4 bytes (metadata format)
            let offset = 1 + 32 + 32 + 4;
            
            // Read name length and name
            const nameLength = data[offset];
            offset += 1;
            const name = data.slice(offset, offset + nameLength).toString('utf8');
            offset += nameLength;
            
            // Read symbol length and symbol
            const symbolLength = data[offset];
            offset += 1;
            const symbol = data.slice(offset, offset + symbolLength).toString('utf8');
            offset += symbolLength;
            
            // Read uri length and uri
            const uriLength = data[offset];
            offset += 1;
            const uri = data.slice(offset, offset + uriLength).toString('utf8');
            
            // Try to fetch token image from URI
            try {
              if (uri) {
                const response = await axios.get(uri);
                
                if (response.data) {
                  if (response.data.description) {
                    metadata.description = response.data.description;
                  }
                  
                  if (response.data.image) {
                    metadata.image = response.data.image;
                  }
                }
              }
            } catch (uriError) {
              logger.debug({ err: uriError }, `Failed to fetch token metadata URI for ${mintPubkey.toString()}`);
            }
            
            // Update metadata
            metadata.name = name;
            metadata.symbol = symbol;
          }
        } catch (error: any) {
          logger.debug({ err: error }, `Failed to get token metadata for ${mintPubkey.toString()}`);
          
          // For well-known tokens, provide hardcoded metadata
          if (mintPubkey.toString() === 'So11111111111111111111111111111111111111112') {
            metadata = {
              name: 'Wrapped SOL',
              symbol: 'SOL',
              description: 'Wrapped SOL is a token that represents SOL in the Solana ecosystem.',
              decimals: 9,
              supply: 0, // Supply is dynamic for wSOL
              image: '',
            };
          }
        }
        
        return metadata;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get token info for ${mintPubkey.toString()}`);
        
        // Return default metadata with decimals from error
        return {
          name: 'Unknown Token',
          symbol: 'UNKNOWN',
          description: '',
          decimals: 0,
          supply: 0,
          image: '',
        };
      }
    }
    
    /**
     * Create a versioned transaction
     * @param instructions - Transaction instructions
     * @param feePayer - Fee payer public key
     * @param lookupTableAddresses - Optional address lookup tables
     * @returns Versioned transaction
     */
    async createVersionedTransaction(
      instructions: Array<any>,
      feePayer: PublicKey,
      lookupTableAddresses: string[] = []
    ): Promise<VersionedTransaction> {
      // Get recent blockhash
      const { blockhash } = await this.getRecentBlockhash();
      
      // Get address lookup tables
      const addressLookupTableAccounts: AddressLookupTableAccount[] = [];
      
      for (const address of lookupTableAddresses) {
        const lookupTable = await this.getLookupTable(address);
        if (lookupTable) {
          addressLookupTableAccounts.push(lookupTable);
        }
      }
      
      // Create transaction message
      const messageV0 = new TransactionMessage({
        payerKey: feePayer,
        recentBlockhash: blockhash,
        instructions,
      }).compileToV0Message(addressLookupTableAccounts);
      
      // Create versioned transaction
      return new VersionedTransaction(messageV0);
    }
    
    /**
     * Sign and send a versioned transaction
     * @param transaction - Versioned transaction
     * @param signers - Transaction signers
     * @returns Transaction signature
     */
    async signAndSendVersionedTransaction(
      transaction: VersionedTransaction,
      signers: Keypair[]
    ): Promise<TransactionSignature> {
      try {
        // Sign the transaction
        transaction.sign(signers);
        
        // Send the transaction
        const signature = await this.connection.sendTransaction(transaction, {
          skipPreflight: this.confirmOptions.skipPreflight,
          preflightCommitment: this.confirmOptions.preflightCommitment,
          maxRetries: this.confirmOptions.maxRetries,
        });
        
        // Confirm the transaction
        await this.connection.confirmTransaction({
          signature,
          ...(await this.getLatestBlockhash()),
        });
        
        return signature;
      } catch (error: any) {
        logger.error({ err: error }, 'Failed to sign and send versioned transaction');
        throw new SolanaTransactionError('Failed to sign and send versioned transaction', error);
      }
    }
    
    /**
     * Get a lookup table
     * @param address - Lookup table address
     * @returns Lookup table account or null if not found
     */
    async getLookupTable(address: string): Promise<AddressLookupTableAccount | null> {
      // Check cache first
      if (this.lookupTableCache.has(address)) {
        return this.lookupTableCache.get(address)!;
      }
      
      try {
        const addressLookupTable = await this.connection.getAddressLookupTable(new PublicKey(address));
        
        if (addressLookupTable?.value) {
          // Cache the lookup table
          this.lookupTableCache.set(address, addressLookupTable.value);
          return addressLookupTable.value;
        }
        
        return null;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get address lookup table ${address}`);
        return null;
      }
    }
    
    /**
     * Get recent blockhash
     * @returns Blockhash and last valid block height
     */
    async getRecentBlockhash(): Promise<{ blockhash: string, lastValidBlockHeight: number }> {
      const key = 'recent';
      
      // Check cache first
      if (this.recentBlockhashes.has(key)) {
        const cached = this.recentBlockhashes.get(key)!;
        
        // Get current block height
        const currentBlockHeight = await this.connection.getBlockHeight();
        
        // Check if cached blockhash is still valid
        if (currentBlockHeight < cached.lastValidBlockHeight) {
          return cached;
        }
      }
      
      // Get new blockhash
      const { blockhash, lastValidBlockHeight } = await this.getLatestBlockhash();
      
      // Cache the blockhash
      this.recentBlockhashes.set(key, { blockhash, lastValidBlockHeight });
      
      return { blockhash, lastValidBlockHeight };
    }
    
    /**
     * Get latest blockhash
     * @returns Blockhash and last valid block height
     */
    async getLatestBlockhash(): Promise<{ blockhash: string, lastValidBlockHeight: number }> {
      return await this.connection.getLatestBlockhash({
        commitment: this.confirmOptions.commitment,
      });
    }
    
    /**
     * Schedule regular blockhash updates
     * @private
     */
    private scheduleBlockhashUpdates(): void {
      // Update blockhash every 30 seconds
      setInterval(async () => {
        try {
          const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash({
            commitment: this.confirmOptions.commitment,
          });
          
          // Cache the blockhash
          this.recentBlockhashes.set('recent', { blockhash, lastValidBlockHeight });
        } catch (error: any) {
          logger.error({ err: error }, 'Failed to update recent blockhash');
        }
      }, 30000);
    }
    
    /**
     * Verify a signature
     * @param signature - Transaction signature
     * @returns True if the transaction was successful
     */
    async verifySignature(signature: string): Promise<boolean> {
      try {
        const { value } = await this.connection.getSignatureStatus(signature);
        
        if (!value) {
          return false;
        }
        
        return value.err === null;
      } catch (error: any) {
        logger.error({ err: error }, `Failed to verify signature ${signature}`);
        return false;
      }
    }
    
    /**
     * Get transaction details
     * @param signature - Transaction signature
     * @returns Transaction details or null if not found
     */
    async getTransaction(signature: string): Promise<ParsedTransactionWithMeta | null> {
      try {
        return await this.connection.getParsedTransaction(
          signature,
          {
            commitment: this.confirmOptions.commitment,
            maxSupportedTransactionVersion: 0,
          }
        );
      } catch (error: any) {
        logger.error({ err: error }, `Failed to get transaction details for ${signature}`);
        return null;
      }
    }
    
    /**
     * Create an associated token account
     * @param wallet - Wallet keypair
     * @param tokenMint - Token mint address
     * @param owner - Optional owner pubkey (defaults to wallet)
     * @returns Transaction signature
     */
    async createAssociatedTokenAccount(
      wallet: Keypair,
      tokenMint: string,
      owner?: string
    ): Promise<string> {
      try {
        const mintPubkey = new PublicKey(tokenMint);
        const ownerPubkey = owner ? new PublicKey(owner) : wallet.publicKey;
        
        // Get associated token account
        const tokenAccount = await getAssociatedTokenAddress(
          mintPubkey,
          ownerPubkey,
          false,
          TOKEN_PROGRAM_ID
        );
        
        // Check if the account already exists
        try {
          await getAccount(this.connection, tokenAccount);
          
          // Account already exists
          return 'Account already exists';
        } catch (error: any) {
          if (!(error instanceof TokenAccountNotFoundError)) {
            throw error;
          }
          
          // Account doesn't exist, create it
          const tx = new Transaction().add(
            createAssociatedTokenAccountInstruction(
              wallet.publicKey, // Payer
              tokenAccount, // Associated token account
              ownerPubkey, // Owner
              mintPubkey, // Mint
              TOKEN_PROGRAM_ID,
              ASSOCIATED_TOKEN_PROGRAM_ID
            )
          );
          
          // Get recent blockhash
          const { blockhash, lastValidBlockHeight } = await this.getRecentBlockhash();
          tx.recentBlockhash = blockhash;
          tx.feePayer = wallet.publicKey;
          
          // Sign and send transaction
          const signature = await sendAndConfirmTransaction(
            this.connection,
            tx,
            [wallet],
            this.confirmOptions
          );
          
          logger.info(`Created associated token account for ${tokenMint}. Signature: ${signature}`);
          
          return signature;
        }
      } catch (error: any) {
        logger.error({ err: error }, `Failed to create associated token account for ${tokenMint}`);
        throw new SolanaTransactionError(`Failed to create associated token account: ${error.message}`, error);
      }
    }
    
    /**
     * Convert a public key to base58 string
     * @param pubkey - Public key
     * @returns Base58 encoded public key
     */
    static publicKeyToString(pubkey: PublicKey): string {
      return pubkey.toString();
    }
    
    /**
     * Convert a base58 string to a public key
     * @param address - Base58 encoded address
     * @returns PublicKey object
     */
    static stringToPublicKey(address: string): PublicKey {
      return new PublicKey(address);
    }
    
    /**
     * Verify if an address is a valid Solana address
     * @param address - Address to verify
     * @returns True if valid
     */
    static isValidAddress(address: string): boolean {
      try {
        new PublicKey(address);
        return true;
      } catch (error) {
        return false;
      }
    }
    
    /**
     * Generate a keypair from a private key
     * @param privateKey - Base58 encoded private key
     * @returns Keypair
     */
    static keypairFromPrivateKey(privateKey: string): Keypair {
      const secretKey = bs58.decode(privateKey);
      return Keypair.fromSecretKey(secretKey);
    }
    
    /**
     * Get the network name for the current connection
     * @returns Network name (mainnet-beta, testnet, devnet, localnet)
     */
    getNetworkName(): string {
      return config.solana.network;
    }
    
    /**
     * Calculate the minimum rent-exempt balance for an account
     * @param size - Account size in bytes
     * @returns Minimum balance in lamports
     */
    async calculateRentExemption(size: number): Promise<number> {
      return await this.connection.getMinimumBalanceForRentExemption(size);
    }
    
    /**
     * Parse error from Solana transaction
     * @param error - Error object
     * @returns Parsed error message
     */
    static parseTransactionError(error: any): string {
      if (error?.logs?.length > 0) {
        // Extract error from logs
        const errorLog = error.logs.find((log: string) => log.includes('Error:'));
        if (errorLog) {
          return errorLog;
        }
        
        // Return all logs
        return error.logs.join('\n');
      }
      
      if (error.message) {
        return error.message;
      }
      
      return 'Unknown transaction error';
    }
  }