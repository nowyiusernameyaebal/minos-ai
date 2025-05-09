/**
 * Vault Processor
 * 
 * Processes vault-related events from the Minos-AI smart contracts
 * and stores them in the database for analysis and reporting.
 * 
 * This processor is responsible for:
 * - Tracking deposits to strategy vaults
 * - Tracking withdrawals from strategy vaults
 * - Recording fee collection events
 * - Monitoring vault balances
 */

import { Connection, ConfirmedSignatureInfo, PublicKey, ParsedTransactionWithMeta, ParsedInstruction } from '@solana/web3.js';
import { DataSource, Repository } from 'typeorm';
import * as anchor from '@project-serum/anchor';
import BN from 'bn.js';
import { BaseProcessor } from './base.processor';
import { VaultTransaction, VaultTransactionType, Strategy, User } from '../entities';
import logger from '../utils/logger';
import { 
  MINOS_AI_AGENT_PROGRAM_ID, 
  SPL_TOKEN_PROGRAM_ID,
  FUNDS_DEPOSITED_EVENT,
  FUNDS_WITHDRAWN_EVENT,
  FEES_COLLECTED_EVENT 
} from '../constants';

export class VaultProcessor extends BaseProcessor {
  private vaultTransactionRepository: Repository<VaultTransaction>;
  private strategyRepository: Repository<Strategy>;
  private userRepository: Repository<User>;
  
  constructor(
    connection: Connection,
    dataSource: DataSource,
    processorName: string = 'VaultProcessor'
  ) {
    super(connection, dataSource, processorName);
    
    this.vaultTransactionRepository = dataSource.getRepository(VaultTransaction);
    this.strategyRepository = dataSource.getRepository(Strategy);
    this.userRepository = dataSource.getRepository(User);
  }
  
  /**
   * Returns the program IDs this processor monitors
   * We monitor both the Minos AI program and SPL Token program
   */
  protected getProgramIds(): PublicKey[] {
    return [
      new PublicKey(MINOS_AI_AGENT_PROGRAM_ID),
      new PublicKey(SPL_TOKEN_PROGRAM_ID)
    ];
  }

  /**
   * Override the base method to handle multiple program IDs
   */
  protected getProgramId(): PublicKey {
    return new PublicKey(MINOS_AI_AGENT_PROGRAM_ID);
  }
  
  /**
   * Process a transaction containing Minos-AI vault-related events
   */
  protected async processTransaction(
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    if (!transaction.meta || transaction.meta.err) {
      return; // Skip failed transactions
    }
    
    // Check for program logs (contains events)
    const logs = transaction.meta.logMessages;
    if (!logs || logs.length === 0) {
      return;
    }
    
    try {
      // Process deposit events
      await this.processDepositEvents(logs, transaction, signature);
      
      // Process withdrawal events
      await this.processWithdrawalEvents(logs, transaction, signature);
      
      // Process fee collection events
      await this.processFeeCollectionEvents(logs, transaction, signature);
      
      // Process SPL Token transfers to track vault balances
      await this.processTokenTransfers(transaction, signature);
      
    } catch (error) {
      logger.error(`Error processing transaction ${signature}:`, error);
    }
  }
  
  /**
   * Process deposit events from transaction logs
   */
  private async processDepositEvents(
    logs: string[],
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract deposit events from logs
    const depositEvents = logs.filter(log => log.includes(FUNDS_DEPOSITED_EVENT));
    
    for (const eventLog of depositEvents) {
      try {
        // Parse the event data from the log
        const eventData = this.parseEventLog(eventLog);
        if (!eventData) continue;
        
        const {
          strategy_id: strategyId,
          depositor,
          timestamp,
          amount
        } = eventData;
        
        // Get or create the related entities
        const strategy = await this.getOrCreateStrategy(strategyId.toString());
        const user = await this.getOrCreateUser(depositor);
        
        // Create vault transaction record
        const vaultTransaction = new VaultTransaction();
        vaultTransaction.strategyId = strategyId.toString();
        vaultTransaction.userAddress = depositor;
        vaultTransaction.timestamp = new Date(timestamp * 1000);
        vaultTransaction.amount = amount;
        vaultTransaction.transactionType = VaultTransactionType.DEPOSIT;
        vaultTransaction.transactionSignature = signature;
        vaultTransaction.strategy = strategy;
        vaultTransaction.user = user;
        
        // Save transaction to database
        await this.vaultTransactionRepository.save(vaultTransaction);
        
        // Update strategy total deposits
        if (strategy) {
          strategy.totalDeposits = (strategy.totalDeposits || 0) + amount;
          await this.strategyRepository.save(strategy);
        }
        
        logger.info(`Indexed vault deposit: strategy=${strategyId}, depositor=${depositor}, amount=${amount}`);
        
      } catch (error) {
        logger.error(`Error processing deposit event:`, error);
      }
    }
  }
  
  /**
   * Process withdrawal events from transaction logs
   */
  private async processWithdrawalEvents(
    logs: string[],
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract withdrawal events from logs
    const withdrawalEvents = logs.filter(log => log.includes(FUNDS_WITHDRAWN_EVENT));
    
    for (const eventLog of withdrawalEvents) {
      try {
        // Parse the event data from the log
        const eventData = this.parseEventLog(eventLog);
        if (!eventData) continue;
        
        const {
          strategy_id: strategyId,
          owner,
          timestamp,
          amount
        } = eventData;
        
        // Get or create the related entities
        const strategy = await this.getOrCreateStrategy(strategyId.toString());
        const user = await this.getOrCreateUser(owner);
        
        // Create vault transaction record
        const vaultTransaction = new VaultTransaction();
        vaultTransaction.strategyId = strategyId.toString();
        vaultTransaction.userAddress = owner;
        vaultTransaction.timestamp = new Date(timestamp * 1000);
        vaultTransaction.amount = amount;
        vaultTransaction.transactionType = VaultTransactionType.WITHDRAWAL;
        vaultTransaction.transactionSignature = signature;
        vaultTransaction.strategy = strategy;
        vaultTransaction.user = user;
        
        // Save transaction to database
        await this.vaultTransactionRepository.save(vaultTransaction);
        
        // Update strategy total withdrawals
        if (strategy) {
          strategy.totalWithdrawals = (strategy.totalWithdrawals || 0) + amount;
          await this.strategyRepository.save(strategy);
        }
        
        logger.info(`Indexed vault withdrawal: strategy=${strategyId}, owner=${owner}, amount=${amount}`);
        
      } catch (error) {
        logger.error(`Error processing withdrawal event:`, error);
      }
    }
  }
  
  /**
   * Process fee collection events from transaction logs
   */
  private async processFeeCollectionEvents(
    logs: string[],
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract fee collection events from logs
    const feeEvents = logs.filter(log => log.includes(FEES_COLLECTED_EVENT));
    
    for (const eventLog of feeEvents) {
      try {
        // Parse the event data from the log
        const eventData = this.parseEventLog(eventLog);
        if (!eventData) continue;
        
        const {
          strategy_id: strategyId,
          collector,
          timestamp,
          performance_fee: performanceFee,
          protocol_fee: protocolFee
        } = eventData;
        
        // Get or create the related entities
        const strategy = await this.getOrCreateStrategy(strategyId.toString());
        const user = await this.getOrCreateUser(collector);
        
        // Create vault transaction record for performance fee
        if (performanceFee > 0) {
          const performanceFeeTransaction = new VaultTransaction();
          performanceFeeTransaction.strategyId = strategyId.toString();
          performanceFeeTransaction.userAddress = collector;
          performanceFeeTransaction.timestamp = new Date(timestamp * 1000);
          performanceFeeTransaction.amount = performanceFee;
          performanceFeeTransaction.transactionType = VaultTransactionType.PERFORMANCE_FEE;
          performanceFeeTransaction.transactionSignature = signature;
          performanceFeeTransaction.strategy = strategy;
          performanceFeeTransaction.user = user;
          
          // Save transaction to database
          await this.vaultTransactionRepository.save(performanceFeeTransaction);
        }
        
        // Create vault transaction record for protocol fee
        if (protocolFee > 0) {
          const protocolFeeTransaction = new VaultTransaction();
          protocolFeeTransaction.strategyId = strategyId.toString();
          protocolFeeTransaction.userAddress = collector;
          protocolFeeTransaction.timestamp = new Date(timestamp * 1000);
          protocolFeeTransaction.amount = protocolFee;
          protocolFeeTransaction.transactionType = VaultTransactionType.PROTOCOL_FEE;
          protocolFeeTransaction.transactionSignature = signature;
          protocolFeeTransaction.strategy = strategy;
          protocolFeeTransaction.user = user;
          
          // Save transaction to database
          await this.vaultTransactionRepository.save(protocolFeeTransaction);
        }
        
        // Update strategy total fees
        if (strategy) {
          strategy.totalFeesCollected = (strategy.totalFeesCollected || 0) + performanceFee + protocolFee;
          await this.strategyRepository.save(strategy);
        }
        
        logger.info(`Indexed fee collection: strategy=${strategyId}, performanceFee=${performanceFee}, protocolFee=${protocolFee}`);
        
      } catch (error) {
        logger.error(`Error processing fee collection event:`, error);
      }
    }
  }
  
  /**
   * Process SPL Token transfers to track vault balances
   */
  private async processTokenTransfers(
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract parsed instructions
    const instructions = transaction.transaction.message.instructions;
    if (!instructions) return;
    
    for (const ix of instructions) {
      // Skip if not an SPL Token instruction
      if (!ix.programId.equals(new PublicKey(SPL_TOKEN_PROGRAM_ID))) continue;
      
      // Skip if not a parsed instruction
      const parsedIx = ix as ParsedInstruction;
      if (!parsedIx.parsed) continue;
      
      // Skip if not a transfer instruction
      if (parsedIx.parsed.type !== 'transfer') continue;
      
      const { info } = parsedIx.parsed;
      
      try {
        // Get strategy from destination (for deposits) or source (for withdrawals)
        const vaultPubkey = info.destination;
        const sourceVaultPubkey = info.source;
        
        // Check if this transfer involves a strategy vault
        const strategyByVault = await this.strategyRepository.findOne({
          where: { vaultAddress: vaultPubkey }
        });
        
        const strategyBySourceVault = await this.strategyRepository.findOne({
          where: { vaultAddress: sourceVaultPubkey }
        });
        
        // Skip if not related to a strategy vault
        if (!strategyByVault && !strategyBySourceVault) continue;
        
        // Process deposit (transfer to vault)
        if (strategyByVault) {
          // Update vault balance (consider calculating balance from token account directly in production)
          strategyByVault.currentBalance = (strategyByVault.currentBalance || 0) + info.amount;
          await this.strategyRepository.save(strategyByVault);
          
          logger.debug(`Updated vault balance for strategy ${strategyByVault.strategyId}: +${info.amount}`);
        }
        
        // Process withdrawal or fee (transfer from vault)
        if (strategyBySourceVault) {
          // Update vault balance (consider calculating balance from token account directly in production)
          strategyBySourceVault.currentBalance = Math.max(0, (strategyBySourceVault.currentBalance || 0) - info.amount);
          await this.strategyRepository.save(strategyBySourceVault);
          
          logger.debug(`Updated vault balance for strategy ${strategyBySourceVault.strategyId}: -${info.amount}`);
        }
        
      } catch (error) {
        logger.error(`Error processing token transfer:`, error);
      }
    }
  }
  
  /**
   * Parse event log into structured data
   */
  private parseEventLog(eventLog: string): any {
    try {
      // Extract JSON data from event log
      const jsonStartIndex = eventLog.indexOf('{');
      if (jsonStartIndex === -1) return null;
      
      const jsonData = eventLog.substring(jsonStartIndex);
      return JSON.parse(jsonData);
    } catch (error) {
      logger.error(`Error parsing event log:`, error);
      return null;
    }
  }
  
  /**
   * Get or create a strategy record
   */
  private async getOrCreateStrategy(strategyId: string): Promise<Strategy> {
    let strategy = await this.strategyRepository.findOne({
      where: { strategyId }
    });
    
    if (!strategy) {
      // Create a placeholder strategy record
      strategy = new Strategy();
      strategy.strategyId = strategyId;
      strategy.createdAt = new Date();
      strategy.updatedAt = new Date();
      strategy.totalDeposits = 0;
      strategy.totalWithdrawals = 0;
      strategy.totalFeesCollected = 0;
      strategy.currentBalance = 0;
      
      await this.strategyRepository.save(strategy);
      logger.debug(`Created placeholder strategy record: ${strategyId}`);
    }
    
    return strategy;
  }
  
  /**
   * Get or create a user record
   */
  private async getOrCreateUser(address: string): Promise<User> {
    let user = await this.userRepository.findOne({
      where: { address }
    });
    
    if (!user) {
      // Create a new user record
      user = new User();
      user.address = address;
      user.firstSeen = new Date();
      user.lastSeen = new Date();
      
      await this.userRepository.save(user);
      logger.debug(`Created new user record: ${address}`);
    } else {
      // Update last seen timestamp
      user.lastSeen = new Date();
      await this.userRepository.save(user);
    }
    
    return user;
  }
}