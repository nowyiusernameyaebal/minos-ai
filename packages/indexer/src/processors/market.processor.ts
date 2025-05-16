/**
 * Market Data Processor for Minos-AI DeFi Platform
 * Processes blockchain data to extract market metrics and insights
 */

import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, Connection } from 'typeorm';
import { Cron } from '@nestjs/schedule';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { ConfigService } from '@nestjs/config';
import { MarketMetrics } from '../entities/market-metrics.entity';
import { ProtocolMetrics } from '../entities/protocol-metrics.entity';
import { PriceData } from '../entities/price-data.entity';
import { TokenTransfer } from '../entities/token-transfer.entity';
import { DefiPosition } from '../entities/defi-position.entity';
import { ProtocolEvent } from '../entities/protocol-event.entity';
import { ethers } from 'ethers';
import * as math from 'mathjs';

export interface MarketProcessorConfig {
  batchSize: number;
  processInterval: number;
  priceUpdateInterval: number;
  enableRealTimeProcessing: boolean;
  protocols: string[];
}

export interface MarketData {
  totalMarketCap: number;
  totalVolume: number;
  totalTvl: number;
  dominanceIndex: number;
  volatilityIndex: number;
  correlationIndex: number;
  liquidityIndex: number;
  timestamp: Date;
}

export interface ProtocolData {
  protocol: string;
  tvl: number;
  volume24h: number;
  fees24h: number;
  transactions24h: number;
  users24h: number;
  apy: number;
  growthRate: number;
  riskScore: number;
  timestamp: Date;
}

export interface PriceMetrics {
  token: string;
  price: number;
  change24h: number;
  volume24h: number;
  volatility: number;
  marketCap: number;
  timestamp: Date;
}

@Injectable()
export class MarketProcessor {
  private readonly logger = new Logger(MarketProcessor.name);
  private isProcessing = false;
  private lastProcessedBlock = 0;

  constructor(
    @InjectRepository(MarketMetrics)
    private marketMetricsRepository: Repository<MarketMetrics>,
    @InjectRepository(ProtocolMetrics)
    private protocolMetricsRepository: Repository<ProtocolMetrics>,
    @InjectRepository(PriceData)
    private priceDataRepository: Repository<PriceData>,
    @InjectRepository(TokenTransfer)
    private tokenTransferRepository: Repository<TokenTransfer>,
    @InjectRepository(DefiPosition)
    private defiPositionRepository: Repository<DefiPosition>,
    @InjectRepository(ProtocolEvent)
    private protocolEventRepository: Repository<ProtocolEvent>,
    private connection: Connection,
    private eventEmitter: EventEmitter2,
    private configService: ConfigService
  ) {
    this.initializeProcessor();
  }

  private async initializeProcessor() {
    try {
      // Get last processed block
      const lastMetrics = await this.marketMetricsRepository.findOne({
        order: { timestamp: 'DESC' }
      });
      
      this.lastProcessedBlock = lastMetrics?.blockNumber || 0;
      this.logger.log(`Market processor initialized at block ${this.lastProcessedBlock}`);
    } catch (error) {
      this.logger.error('Failed to initialize market processor', error);
    }
  }

  @Cron('*/5 * * * *') // Every 5 minutes
  async processMarketData() {
    if (this.isProcessing) {
      this.logger.warn('Market processing already in progress, skipping...');
      return;
    }

    this.isProcessing = true;
    
    try {
      this.logger.log('Starting market data processing...');
      
      // Process market metrics
      await this.processMarketMetrics();
      
      // Process protocol metrics
      await this.processProtocolMetrics();
      
      // Process price data
      await this.processPriceData();
      
      // Calculate derived metrics
      await this.calculateDerivedMetrics();
      
      this.logger.log('Market data processing completed successfully');
      
      // Emit event for real-time updates
      this.eventEmitter.emit('market.updated', {
        timestamp: new Date(),
        processedBlock: this.lastProcessedBlock
      });
      
    } catch (error) {
      this.logger.error('Error processing market data', error);
    } finally {
      this.isProcessing = false;
    }
  }

  private async processMarketMetrics() {
    const startTime = Date.now();
    
    try {
      // Calculate total market metrics
      const marketData = await this.calculateMarketMetrics();
      
      // Save market metrics
      const marketMetrics = this.marketMetricsRepository.create({
        ...marketData,
        blockNumber: this.lastProcessedBlock
      });
      
      await this.marketMetricsRepository.save(marketMetrics);
      
      this.logger.log(`Market metrics processed in ${Date.now() - startTime}ms`);
    } catch (error) {
      this.logger.error('Error processing market metrics', error);
      throw error;
    }
  }

  private async calculateMarketMetrics(): Promise<MarketData> {
    const timestamp = new Date();
    const yesterday = new Date(timestamp.getTime() - 24 * 60 * 60 * 1000);
    
    // Get all active protocols
    const protocols = await this.protocolMetricsRepository
      .createQueryBuilder('pm')
      .select('DISTINCT(pm.protocol)', 'protocol')
      .getRawMany();
    
    let totalMarketCap = 0;
    let totalVolume = 0;
    let totalTvl = 0;
    let protocolData: any[] = [];
    
    // Calculate metrics for each protocol
    for (const { protocol } of protocols) {
      const latestMetrics = await this.protocolMetricsRepository.findOne({
        where: { protocol },
        order: { timestamp: 'DESC' }
      });
      
      if (latestMetrics) {
        totalMarketCap += latestMetrics.marketCap || 0;
        totalVolume += latestMetrics.volume24h || 0;
        totalTvl += latestMetrics.tvl || 0;
        protocolData.push(latestMetrics);
      }
    }
    
    // Calculate dominance index (top protocol's market share)
    const dominanceIndex = protocolData.length > 0 
      ? Math.max(...protocolData.map(p => p.marketCap || 0)) / totalMarketCap * 100
      : 0;
    
    // Calculate volatility index
    const volatilityIndex = await this.calculateVolatilityIndex();
    
    // Calculate correlation index
    const correlationIndex = await this.calculateCorrelationIndex();
    
    // Calculate liquidity index
    const liquidityIndex = await this.calculateLiquidityIndex();
    
    return {
      totalMarketCap,
      totalVolume,
      totalTvl,
      dominanceIndex,
      volatilityIndex,
      correlationIndex,
      liquidityIndex,
      timestamp
    };
  }

  private async processProtocolMetrics() {
    const startTime = Date.now();
    const protocols = this.configService.get<string[]>('market.protocols', []);
    
    try {
      for (const protocol of protocols) {
        await this.processProtocolData(protocol);
      }
      
      this.logger.log(`Protocol metrics processed in ${Date.now() - startTime}ms`);
    } catch (error) {
      this.logger.error('Error processing protocol metrics', error);
      throw error;
    }
  }

  private async processProtocolData(protocol: string) {
    const timestamp = new Date();
    const yesterday = new Date(timestamp.getTime() - 24 * 60 * 60 * 1000);
    
    // Calculate TVL
    const tvl = await this.defiPositionRepository
      .createQueryBuilder('position')
      .where('position.protocol = :protocol', { protocol })
      .andWhere('position.isActive = true')
      .select('SUM(position.value)', 'total')
      .getRawOne();
    
    // Calculate 24h volume
    const volume24h = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.protocol = :protocol', { protocol })
      .andWhere('event.timestamp >= :yesterday', { yesterday })
      .andWhere('event.eventType IN (:...types)', { types: ['swap', 'add_liquidity', 'remove_liquidity'] })
      .select('SUM(CAST(event.metadata->>"value" AS DECIMAL))', 'total')
      .getRawOne();
    
    // Calculate 24h fees
    const fees24h = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.protocol = :protocol', { protocol })
      .andWhere('event.timestamp >= :yesterday', { yesterday })
      .andWhere('event.eventType = :type', { type: 'fee_collection' })
      .select('SUM(CAST(event.metadata->>"feeAmount" AS DECIMAL))', 'total')
      .getRawOne();
    
    // Calculate 24h transactions
    const transactions24h = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.protocol = :protocol', { protocol })
      .andWhere('event.timestamp >= :yesterday', { yesterday })
      .select('COUNT(*)', 'total')
      .getRawOne();
    
    // Calculate unique users in 24h
    const users24h = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.protocol = :protocol', { protocol })
      .andWhere('event.timestamp >= :yesterday', { yesterday })
      .select('COUNT(DISTINCT event.userAddress)', 'total')
      .getRawOne();
    
    // Calculate growth rate (7-day)
    const growthRate = await this.calculateGrowthRate(protocol);
    
    // Calculate APY
    const apy = await this.calculateProtocolAPY(protocol);
    
    // Calculate risk score
    const riskScore = await this.calculateRiskScore(protocol);
    
    // Create protocol metrics
    const protocolMetrics = this.protocolMetricsRepository.create({
      protocol,
      tvl: parseFloat(tvl?.total || '0'),
      volume24h: parseFloat(volume24h?.total || '0'),
      fees24h: parseFloat(fees24h?.total || '0'),
      transactions24h: parseInt(transactions24h?.total || '0'),
      users24h: parseInt(users24h?.total || '0'),
      apy,
      growthRate,
      riskScore,
      timestamp,
      blockNumber: this.lastProcessedBlock
    });
    
    await this.protocolMetricsRepository.save(protocolMetrics);
  }

  private async processPriceData() {
    const startTime = Date.now();
    
    try {
      // Get all unique tokens
      const tokens = await this.tokenTransferRepository
        .createQueryBuilder('transfer')
        .select('DISTINCT(transfer.tokenAddress)', 'token')
        .getRawMany();
      
      for (const { token } of tokens) {
        await this.processPriceMetrics(token);
      }
      
      this.logger.log(`Price data processed in ${Date.now() - startTime}ms`);
    } catch (error) {
      this.logger.error('Error processing price data', error);
      throw error;
    }
  }

  private async processPriceMetrics(tokenAddress: string) {
    const timestamp = new Date();
    const yesterday = new Date(timestamp.getTime() - 24 * 60 * 60 * 1000);
    
    // Get current price (this would typically come from an oracle)
    const currentPrice = await this.getCurrentPrice(tokenAddress);
    
    // Get price 24h ago
    const price24hAgo = await this.priceDataRepository.findOne({
      where: { 
        tokenAddress,
        timestamp: this.connection.createQueryBuilder()
          .where('timestamp <= :yesterday', { yesterday })
          .orderBy('timestamp', 'DESC')
      }
    });
    
    // Calculate 24h change
    const change24h = price24hAgo 
      ? ((currentPrice - price24hAgo.price) / price24hAgo.price) * 100
      : 0;
    
    // Calculate 24h volume
    const volume24h = await this.tokenTransferRepository
      .createQueryBuilder('transfer')
      .where('transfer.tokenAddress = :tokenAddress', { tokenAddress })
      .andWhere('transfer.timestamp >= :yesterday', { yesterday })
      .select('SUM(CAST(transfer.value AS DECIMAL) * :price)', { price: currentPrice })
      .getRawOne();
    
    // Calculate volatility (7-day)
    const volatility = await this.calculatePriceVolatility(tokenAddress);
    
    // Get market cap (would require token supply data)
    const marketCap = await this.calculateMarketCap(tokenAddress, currentPrice);
    
    // Save price data
    const priceData = this.priceDataRepository.create({
      tokenAddress,
      price: currentPrice,
      change24h,
      volume24h: parseFloat(volume24h?.sum || '0'),
      volatility,
      marketCap,
      timestamp,
      blockNumber: this.lastProcessedBlock
    });
    
    await this.priceDataRepository.save(priceData);
  }

  private async calculateDerivedMetrics() {
    // Calculate correlations between protocols
    await this.calculateProtocolCorrelations();
    
    // Update yield metrics
    await this.updateYieldMetrics();
    
    // Calculate risk-adjusted returns
    await this.calculateRiskAdjustedReturns();
    
    // Update liquidity metrics
    await this.updateLiquidityMetrics();
  }

  private async calculateVolatilityIndex(): Promise<number> {
    const protocols = await this.protocolMetricsRepository
      .createQueryBuilder('pm')
      .select('DISTINCT(pm.protocol)', 'protocol')
      .getRawMany();
    
    let totalVolatility = 0;
    let count = 0;
    
    for (const { protocol } of protocols) {
      const metrics = await this.protocolMetricsRepository.find({
        where: { protocol },
        order: { timestamp: 'DESC' },
        take: 30 // Last 30 data points
      });
      
      if (metrics.length >= 10) {
        const values = metrics.map(m => m.tvl);
        const volatility = this.calculateVolatility(values);
        totalVolatility += volatility;
        count++;
      }
    }
    
    return count > 0 ? totalVolatility / count : 0;
  }

  private async calculateCorrelationIndex(): Promise<number> {
    // Simplified correlation index calculation
    // In practice, this would calculate correlations between protocol movements
    return 0.5; // Placeholder
  }

  private async calculateLiquidityIndex(): Promise<number> {
    const totalTvl = await this.defiPositionRepository
      .createQueryBuilder('position')
      .where('position.isActive = true')
      .select('SUM(position.value)', 'total')
      .getRawOne();
    
    const totalVolume = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.timestamp >= :yesterday', { yesterday: new Date(Date.now() - 24*60*60*1000) })
      .andWhere('event.eventType IN (:...types)', { types: ['swap', 'add_liquidity', 'remove_liquidity'] })
      .select('SUM(CAST(event.metadata->>"value" AS DECIMAL))', 'total')
      .getRawOne();
    
    const tvl = parseFloat(totalTvl?.total || '0');
    const volume = parseFloat(totalVolume?.total || '0');
    
    // Liquidity index = volume/tvl ratio normalized to 0-100
    return tvl > 0 ? Math.min((volume / tvl) * 100, 100) : 0;
  }

  private async calculateGrowthRate(protocol: string): Promise<number> {
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    
    const currentMetrics = await this.protocolMetricsRepository.findOne({
      where: { protocol },
      order: { timestamp: 'DESC' }
    });
    
    const weekAgoMetrics = await this.protocolMetricsRepository.findOne({
      where: { 
        protocol,
        timestamp: this.connection.createQueryBuilder()
          .where('timestamp <= :weekAgo', { weekAgo })
          .orderBy('timestamp', 'DESC')
      }
    });
    
    if (!currentMetrics || !weekAgoMetrics || weekAgoMetrics.tvl === 0) {
      return 0;
    }
    
    return ((currentMetrics.tvl - weekAgoMetrics.tvl) / weekAgoMetrics.tvl) * 100;
  }

  private async calculateProtocolAPY(protocol: string): Promise<number> {
    // This would typically calculate based on yield farming rewards
    // Simplified version returning average yield
    const yieldEvents = await this.protocolEventRepository
      .createQueryBuilder('event')
      .where('event.protocol = :protocol', { protocol })
      .andWhere('event.eventType = :type', { type: 'yield_distributed' })
      .andWhere('event.timestamp >= :yesterday', { yesterday: new Date(Date.now() - 24*60*60*1000) })
      .getMany();
    
    // Calculate APY based on yield events
    // This is a simplified calculation
    return 5.0; // Placeholder APY
  }

  private async calculateRiskScore(protocol: string): Promise<number> {
    // Calculate risk based on multiple factors
    let riskScore = 0;
    
    // Volatility risk
    const metrics = await this.protocolMetricsRepository.find({
      where: { protocol },
      order: { timestamp: 'DESC' },
      take: 30
    });
    
    if (metrics.length >= 10) {
      const values = metrics.map(m => m.tvl);
      const volatility = this.calculateVolatility(values);
      riskScore += Math.min(volatility * 10, 30); // Max 30% from volatility
    }
    
    // Concentration risk
    const totalUsers = await this.defiPositionRepository
      .createQueryBuilder('position')
      .where('position.protocol = :protocol', { protocol })
      .select('COUNT(DISTINCT position.userAddress)', 'count')
      .getRawOne();
    
    const userCount = parseInt(totalUsers?.count || '0');
    if (userCount < 100) {
      riskScore += 20; // Add risk for low user count
    }
    
    // Smart contract risk (simplified)
    riskScore += 10; // Base smart contract risk
    
    return Math.min(riskScore, 100); // Cap at 100
  }

  private calculateVolatility(values: number[]): number {
    if (values.length < 2) return 0;
    
    const returns = values.slice(1).map((value, i) => {
      const prevValue = values[i];
      return prevValue !== 0 ? (value - prevValue) / prevValue : 0;
    });
    
    return math.std(returns);
  }

  private async getCurrentPrice(tokenAddress: string): Promise<number> {
    // This would integrate with price oracles like Chainlink
    // For now, return a mock price
    return Math.random() * 1000;
  }

  private async calculatePriceVolatility(tokenAddress: string): Promise<number> {
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    
    const prices = await this.priceDataRepository.find({
      where: { 
        tokenAddress,
        timestamp: this.connection.createQueryBuilder()
          .where('timestamp >= :weekAgo', { weekAgo })
      },
      order: { timestamp: 'ASC' }
    });
    
    if (prices.length < 2) return 0;
    
    const returns = prices.slice(1).map((price, i) => {
      const prevPrice = prices[i].price;
      return prevPrice !== 0 ? (price.price - prevPrice) / prevPrice : 0;
    });
    
    return math.std(returns);
  }

  private async calculateMarketCap(tokenAddress: string, price: number): Promise<number> {
    // This would require token supply data
    // Simplified calculation
    return price * 1000000; // Assuming 1M supply
  }

  private async calculateProtocolCorrelations() {
    const protocols = await this.protocolMetricsRepository
      .createQueryBuilder('pm')
      .select('DISTINCT(pm.protocol)', 'protocol')
      .getRawMany();
    
    // Calculate pairwise correlations
    for (let i = 0; i < protocols.length; i++) {
      for (let j = i + 1; j < protocols.length; j++) {
        const correlation = await this.calculatePairwiseCorrelation(
          protocols[i].protocol,
          protocols[j].protocol
        );
        
        // Store correlation data (would need a correlation table)
        this.logger.debug(`Correlation between ${protocols[i].protocol} and ${protocols[j].protocol}: ${correlation}`);
      }
    }
  }

  private async calculatePairwiseCorrelation(protocol1: string, protocol2: string): Promise<number> {
    const metrics1 = await this.protocolMetricsRepository.find({
      where: { protocol: protocol1 },
      order: { timestamp: 'DESC' },
      take: 30
    });
    
    const metrics2 = await this.protocolMetricsRepository.find({
      where: { protocol: protocol2 },
      order: { timestamp: 'DESC' },
      take: 30
    });
    
    if (metrics1.length !== metrics2.length || metrics1.length < 10) {
      return 0;
    }
    
    const values1 = metrics1.map(m => m.tvl);
    const values2 = metrics2.map(m => m.tvl);
    
    return this.calculateCorrelation(values1, values2);
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator !== 0 ? numerator / denominator : 0;
  }

  private async updateYieldMetrics() {
    // Update yield-related metrics
    this.logger.debug('Updating yield metrics...');
  }

  private async calculateRiskAdjustedReturns() {
    // Calculate Sharpe ratios and other risk-adjusted metrics
    this.logger.debug('Calculating risk-adjusted returns...');
  }

  private async updateLiquidityMetrics() {
    // Update liquidity-related metrics
    this.logger.debug('Updating liquidity metrics...');
  }

  async getLatestMarketMetrics(): Promise<MarketMetrics | null> {
    return this.marketMetricsRepository.findOne({
      order: { timestamp: 'DESC' }
    });
  }

  async getProtocolMetrics(protocol: string): Promise<ProtocolMetrics[]> {
    return this.protocolMetricsRepository.find({
      where: { protocol },
      order: { timestamp: 'DESC' },
      take: 100
    });
  }

  async getPriceData(tokenAddress: string): Promise<PriceData[]> {
    return this.priceDataRepository.find({
      where: { tokenAddress },
      order: { timestamp: 'DESC' },
      take: 100
    });
  }
}