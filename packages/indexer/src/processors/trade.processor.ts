/**
 * Trade Processor
 * 
 * Processes trade-related events from the Minos-AI smart contracts
 * and stores them in the database for analysis and reporting.
 * 
 * This processor is responsible for:
 * - Indexing signal submissions
 * - Tracking trade executions
 * - Recording performance metrics
 */

import { Connection, ConfirmedSignatureInfo, PublicKey, ParsedTransactionWithMeta } from '@solana/web3.js';
import { DataSource, Repository } from 'typeorm';
import * as anchor from '@project-serum/anchor';
import BN from 'bn.js';
import { BaseProcessor } from './base.processor';
import { Trade, Signal, TradeStatus, Model, Strategy } from '../entities';
import logger from '../utils/logger';
import { MINOS_AI_AGENT_PROGRAM_ID, SIGNAL_SUBMITTED_EVENT, SIGNAL_EXECUTED_EVENT } from '../constants';

export class TradeProcessor extends BaseProcessor {
  private tradeRepository: Repository<Trade>;
  private signalRepository: Repository<Signal>;
  private modelRepository: Repository<Model>;
  private strategyRepository: Repository<Strategy>;
  
  constructor(
    connection: Connection,
    dataSource: DataSource,
    processorName: string = 'TradeProcessor'
  ) {
    super(connection, dataSource, processorName);
    
    this.tradeRepository = dataSource.getRepository(Trade);
    this.signalRepository = dataSource.getRepository(Signal);
    this.modelRepository = dataSource.getRepository(Model);
    this.strategyRepository = dataSource.getRepository(Strategy);
  }
  
  /**
   * Returns the program ID this processor monitors
   */
  protected getProgramId(): PublicKey {
    return new PublicKey(MINOS_AI_AGENT_PROGRAM_ID);
  }
  
  /**
   * Process a transaction containing Minos-AI trade-related events
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
      // Process signal submission events
      await this.processSignalSubmissionEvents(logs, transaction, signature);
      
      // Process signal execution events
      await this.processSignalExecutionEvents(logs, transaction, signature);
      
    } catch (error) {
      logger.error(`Error processing transaction ${signature}:`, error);
    }
  }
  
  /**
   * Process signal submission events from transaction logs
   */
  private async processSignalSubmissionEvents(
    logs: string[],
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract signal submission events from logs
    const signalSubmittedEvents = logs.filter(log => log.includes(SIGNAL_SUBMITTED_EVENT));
    
    for (const eventLog of signalSubmittedEvents) {
      try {
        // Parse the event data from the log
        const eventData = this.parseEventLog(eventLog);
        if (!eventData) continue;
        
        const {
          model_id: modelId,
          signal_id: signalId,
          strategy_id: strategyId,
          submitter,
          timestamp,
          signal_type: signalType,
          verified
        } = eventData;
        
        // Check if signal already exists in database
        const existingSignal = await this.signalRepository.findOne({
          where: {
            modelId: modelId.toString(),
            signalId: signalId.toString()
          }
        });
        
        if (existingSignal) {
          logger.debug(`Signal ${modelId}-${signalId} already exists in database, skipping`);
          continue;
        }
        
        // Get related model and strategy
        const model = await this.modelRepository.findOne({
          where: { modelId: modelId.toString() }
        });
        
        const strategy = await this.strategyRepository.findOne({
          where: { strategyId: strategyId.toString() }
        });
        
        // Create new signal record
        const signal = new Signal();
        signal.modelId = modelId.toString();
        signal.signalId = signalId.toString();
        signal.strategyId = strategyId.toString();
        signal.submitter = submitter;
        signal.timestamp = new Date(timestamp * 1000);
        signal.signalType = signalType;
        signal.verified = verified;
        signal.transactionSignature = signature;
        signal.model = model;
        signal.strategy = strategy;
        
        // Save signal to database
        await this.signalRepository.save(signal);
        logger.info(`Indexed signal submission: model=${modelId}, signal=${signalId}, strategy=${strategyId}`);
        
      } catch (error) {
        logger.error(`Error processing signal submission event:`, error);
      }
    }
  }
  
  /**
   * Process signal execution events from transaction logs
   */
  private async processSignalExecutionEvents(
    logs: string[],
    transaction: ParsedTransactionWithMeta,
    signature: string
  ): Promise<void> {
    // Extract signal execution events from logs
    const signalExecutedEvents = logs.filter(log => log.includes(SIGNAL_EXECUTED_EVENT));
    
    for (const eventLog of signalExecutedEvents) {
      try {
        // Parse the event data from the log
        const eventData = this.parseEventLog(eventLog);
        if (!eventData) continue;
        
        const {
          model_id: modelId,
          signal_id: signalId,
          strategy_id: strategyId,
          executor,
          timestamp,
          execution_result: executionResult,
          price
        } = eventData;
        
        // Find the signal in the database
        const signal = await this.signalRepository.findOne({
          where: {
            modelId: modelId.toString(),
            signalId: signalId.toString()
          }
        });
        
        if (!signal) {
          logger.warn(`Signal ${modelId}-${signalId} not found in database, creating placeholder`);
          // Create placeholder signal (this shouldn't happen in normal circumstances)
          const newSignal = new Signal();
          newSignal.modelId = modelId.toString();
          newSignal.signalId = signalId.toString();
          newSignal.strategyId = strategyId.toString();
          newSignal.timestamp = new Date(timestamp * 1000);
          newSignal.transactionSignature = signature;
          await this.signalRepository.save(newSignal);
        }
        
        // Get related model and strategy
        const model = await this.modelRepository.findOne({
          where: { modelId: modelId.toString() }
        });
        
        const strategy = await this.strategyRepository.findOne({
          where: { strategyId: strategyId.toString() }
        });
        
        // Create trade record
        const trade = new Trade();
        trade.modelId = modelId.toString();
        trade.signalId = signalId.toString();
        trade.strategyId = strategyId.toString();
        trade.executor = executor;
        trade.timestamp = new Date(timestamp * 1000);
        trade.executionResult = executionResult;
        trade.executionPrice = price;
        trade.status = executionResult > 0 ? TradeStatus.PROFIT : (executionResult < 0 ? TradeStatus.LOSS : TradeStatus.NEUTRAL);
        trade.transactionSignature = signature;
        trade.signal = signal;
        trade.model = model;
        trade.strategy = strategy;
        
        // Save trade to database
        await this.tradeRepository.save(trade);
        
        // Update signal as executed
        if (signal) {
          signal.executed = true;
          signal.executionTime = new Date(timestamp * 1000);
          signal.executionResult = executionResult;
          await this.signalRepository.save(signal);
        }
        
        logger.info(`Indexed trade execution: model=${modelId}, signal=${signalId}, strategy=${strategyId}, result=${executionResult}`);
        
      } catch (error) {
        logger.error(`Error processing signal execution event:`, error);
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
   * Utility function to convert BN to number
   */
  private bnToNumber(bn: BN): number {
    return bn.toNumber();
  }
}