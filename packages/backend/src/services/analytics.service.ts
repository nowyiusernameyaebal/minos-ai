/**
 * Analytics Service for Minos-AI DeFi Platform
 * Comprehensive analytics and metrics service for DeFi portfolios and protocols
 */

import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, Connection } from 'typeorm';
import { Cron } from '@nestjs/schedule';
import { 
  PortfolioAnalytics, 
  ProtocolMetrics, 
  MarketMetrics,
  PerformanceMetrics,
  RiskMetrics,
  YieldMetrics,
  LiquidityMetrics,
  SocialSentimentMetrics,
  TrendAnalysis,
  AlertMetrics
} from '../entities/analytics.entity';
import { Portfolio } from '../entities/portfolio.entity';
import { ProtocolEvent } from '../entities/protocol-event.entity';
import { TokenTransfer } from '../entities/token-transfer.entity';
import { DefiPosition } from '../entities/defi-position.entity';
import { Alert } from '../entities/alert.entity';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { Redis } from 'ioredis';
import { ConfigService } from '@nestjs/config';
import { 
  startOfDay, 
  endOfDay, 
  subDays, 
  subMonths,
  differenceInDays 
} from 'date-fns';
import { ethers } from 'ethers';
import * as math from 'mathjs';

export interface TimeRange {
  from: Date;
  to: Date;
}

export interface AnalyticsConfig {
  cacheEnabled: boolean;
  cacheTtl: number;
  batchSize: number;
  enableRealTimeUpdates: boolean;
}

export interface PortfolioAnalyticsData {
  totalValue: number;
  totalReturn: number;
  totalReturnPercentage: number;
  totalGain: number;
  totalGainPercentage: number;
  sharpeRatio: number;
  drawdown: number;
  volatility: number;
  alpha: number;
  beta: number;
  performanceHistory: Array<{
    date: Date;
    value: number;
    return: number;
  }>;
  assetAllocation: Array<{
    protocol: string;
    allocation: number;
    value: number;
  }>;
  riskMetrics: {
    var95: number;
    var99: number;
    expectedShortfall: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
}

export interface ProtocolAnalyticsData {
  tvl: number;
  volume24h: number;
  fees24h: number;
  transactions24h: number;
  apy: number;
  users: number;
  growthRate: number;
  marketShare: number;
  riskScore: number;
  liquidityScore: number;
  trends: {
    tvlGrowth: number;
    volumeGrowth: number;
    userGrowth: number;
  };
}

export interface MarketAnalyticsData {
  totalMarketCap: number;
  totalVolume: number;
  dominanceIndex: number;
  volatilityIndex: number;
  sentimentScore: number;
  fearGreedIndex: number;
  correlationMatrix: Record<string, Record<string, number>>;
  sectorPerformance: Array<{
    sector: string;
    performance: number;
    volume: number;
  }>;
}

@Injectable()
export class AnalyticsService {
  private readonly logger = new Logger(AnalyticsService.name);

  constructor(
    @InjectRepository(PortfolioAnalytics)
    private portfolioAnalyticsRepository: Repository<PortfolioAnalytics>,
    @InjectRepository(ProtocolMetrics)
    private protocolMetricsRepository: Repository<ProtocolMetrics>,
    @InjectRepository(MarketMetrics)
    private marketMetricsRepository: Repository<MarketMetrics>,
    @InjectRepository(Portfolio)
    private portfolioRepository: Repository<Portfolio>,
    @InjectRepository(ProtocolEvent)
    private protocolEventRepository: Repository<ProtocolEvent>,
    @InjectRepository(TokenTransfer)
    private tokenTransferRepository: Repository<TokenTransfer>,
    @InjectRepository(DefiPosition)
    private defiPositionRepository: Repository<DefiPosition>,
    @InjectRepository(Alert)
    private alertRepository: Repository<Alert>,
    private connection: Connection,
    private eventEmitter: EventEmitter2,
    private configService: ConfigService,
    private readonly redis: Redis
  ) {}

  async getPortfolioAnalytics(
    portfolioId: string, 
    timeRange: TimeRange
  ): Promise<PortfolioAnalyticsData> {
    const cacheKey = `analytics:portfolio:${portfolioId}:${timeRange.from.getTime()}:${timeRange.to.getTime()}`;
    
    // Check cache first
    if (this.configService.get('analytics.cacheEnabled')) {
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    const analytics = await this.calculatePortfolioAnalytics(portfolioId, timeRange);
    
    // Cache the result
    await this.redis.setex(
      cacheKey, 
      this.configService.get('analytics.cacheTtl', 3600),
      JSON.stringify(analytics)
    );

    return analytics;
  }

  private async calculatePortfolioAnalytics(
    portfolioId: string, 
    timeRange: TimeRange
  ): Promise<PortfolioAnalyticsData> {
    // Get portfolio data
    const portfolio = await this.portfolioRepository.findOne({
      where: { id: portfolioId },
      relations: ['positions', 'allocations']
    });

    if (!portfolio) {
      throw new Error(`Portfolio ${portfolioId} not found`);
    }

    // Calculate current portfolio value
    const currentValue = await this.calculatePortfolioValue(portfolio);
    
    // Get historical data
    const historicalData = await this.getPortfolioHistoricalData(portfolioId, timeRange);
    
    // Calculate returns
    const initialValue = historicalData.length > 0 ? historicalData[0].value : currentValue;
    const totalReturn = currentValue - initialValue;
    const totalReturnPercentage = initialValue > 0 ? (totalReturn / initialValue) * 100 : 0;

    // Calculate performance metrics
    const returns = historicalData.map((data, index) => {
      if (index === 0) return 0;
      const prevValue = historicalData[index - 1].value;
      return prevValue > 0 ? (data.value - prevValue) / prevValue : 0;
    });

    const sharpeRatio = this.calculateSharpeRatio(returns);
    const drawdown = this.calculateMaxDrawdown(historicalData.map(d => d.value));
    const volatility = this.calculateVolatility(returns);
    
    // Calculate alpha and beta (against market benchmark)
    const marketReturns = await this.getMarketBenchmarkReturns(timeRange);
    const { alpha, beta } = this.calculateAlphaBeta(returns, marketReturns);

    // Get asset allocation
    const assetAllocation = await this.getAssetAllocation(portfolio);

    // Calculate risk metrics
    const riskMetrics = this.calculateRiskMetrics(returns);

    return {
      totalValue: currentValue,
      totalReturn,
      totalReturnPercentage,
      totalGain: totalReturn,
      totalGainPercentage: totalReturnPercentage,
      sharpeRatio,
      drawdown,
      volatility,
      alpha,
      beta,
      performanceHistory: historicalData,
      assetAllocation,
      riskMetrics
    };
  }

  async getProtocolAnalytics(protocol: string, timeRange: TimeRange): Promise<ProtocolAnalyticsData> {
    const cacheKey = `analytics:protocol:${protocol}:${timeRange.from.getTime()}:${timeRange.to.getTime()}`;
    
    // Check cache
    if (this.configService.get('analytics.cacheEnabled')) {
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    const analytics = await this.calculateProtocolAnalytics(protocol, timeRange);
    
    // Cache the result
    await this.redis.setex(
      cacheKey,
      this.configService.get('analytics.cacheTtl', 3600),
      JSON.stringify(analytics)
    );

    return analytics;
  }

  private async calculateProtocolAnalytics(
    protocol: string,
    timeRange: TimeRange
  ): Promise<ProtocolAnalyticsData> {
    // Get latest protocol metrics
    const latestMetrics = await this.protocolMetricsRepository.findOne({
      where: { protocol },
      order: { timestamp: 'DESC' }
    });

    if (!latestMetrics) {
      throw new Error(`No metrics found for protocol ${protocol}`);
    }

    // Calculate 24h metrics
    const yesterday = subDays(new Date(), 1);
    const metrics24hAgo = await this.protocolMetricsRepository.findOne({
      where: { protocol, timestamp: this.connection.createQueryBuilder().where('timestamp <= :date', { date: yesterday }) },
      order: { timestamp: 'DESC' }
    });

    // Calculate growth rates
    const tvlGrowth = metrics24hAgo 
      ? ((latestMetrics.tvl - metrics24hAgo.tvl) / metrics24hAgo.tvl) * 100
      : 0;

    const volumeGrowth = metrics24hAgo
      ? ((latestMetrics.volume24h - metrics24hAgo.volume24h) / metrics24hAgo.volume24h) * 100
      : 0;

    // Get user count
    const uniqueUsers = await this.defiPositionRepository
      .createQueryBuilder('position')
      .where('position.protocol = :protocol', { protocol })
      .andWhere('position.isActive = true')
      .select('COUNT(DISTINCT position.userAddress)', 'count')
      .getRawOne();

    // Calculate user growth
    const userGrowth = await this.calculateUserGrowth(protocol, timeRange);

    // Calculate market share
    const totalMarketTvl = await this.getTotalMarketTvl();
    const marketShare = totalMarketTvl > 0 ? (latestMetrics.tvl / totalMarketTvl) * 100 : 0;

    // Calculate risk score
    const riskScore = await this.calculateProtocolRiskScore(protocol);

    // Calculate liquidity score
    const liquidityScore = await this.calculateLiquidityScore(protocol);

    return {
      tvl: latestMetrics.tvl,
      volume24h: latestMetrics.volume24h,
      fees24h: latestMetrics.fees24h,
      transactions24h: latestMetrics.transactions24h,
      apy: latestMetrics.apy || 0,
      users: parseInt(uniqueUsers.count || '0'),
      growthRate: tvlGrowth,
      marketShare,
      riskScore,
      liquidityScore,
      trends: {
        tvlGrowth,
        volumeGrowth,
        userGrowth
      }
    };
  }

  async getMarketAnalytics(): Promise<MarketAnalyticsData> {
    const cacheKey = 'analytics:market:latest';
    
    // Check cache
    if (this.configService.get('analytics.cacheEnabled')) {
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    const analytics = await this.calculateMarketAnalytics();
    
    // Cache the result
    await this.redis.setex(cacheKey, 300, JSON.stringify(analytics)); // 5 min cache

    return analytics;
  }

  private async calculateMarketAnalytics(): Promise<MarketAnalyticsData> {
    // Get latest market metrics
    const latestMetrics = await this.marketMetricsRepository.findOne({
      order: { timestamp: 'DESC' }
    });

    if (!latestMetrics) {
      throw new Error('No market metrics found');
    }

    // Calculate correlation matrix
    const correlationMatrix = await this.calculateCorrelationMatrix();

    // Get sector performance
    const sectorPerformance = await this.getSectorPerformance();

    return {
      totalMarketCap: latestMetrics.totalMarketCap,
      totalVolume: latestMetrics.totalVolume,
      dominanceIndex: latestMetrics.dominanceIndex,
      volatilityIndex: latestMetrics.volatilityIndex,
      sentimentScore: latestMetrics.sentimentScore,
      fearGreedIndex: latestMetrics.fearGreedIndex,
      correlationMatrix,
      sectorPerformance
    };
  }

  async getRiskAnalytics(portfolioId: string): Promise<RiskMetrics> {
    const portfolio = await this.portfolioRepository.findOne({
      where: { id: portfolioId },
      relations: ['positions']
    });

    if (!portfolio) {
      throw new Error(`Portfolio ${portfolioId} not found`);
    }

    // Calculate VaR (Value at Risk)
    const var95 = await this.calculateVaR(portfolioId, 0.95);
    const var99 = await this.calculateVaR(portfolioId, 0.99);

    // Calculate Expected Shortfall
    const expectedShortfall = await this.calculateExpectedShortfall(portfolioId, 0.95);

    // Calculate maximum drawdown
    const maxDrawdown = await this.calculateMaxDrawdown(portfolioId);

    // Calculate beta against market
    const beta = await this.calculatePortfolioBeta(portfolioId);

    // Calculate concentration risk
    const concentrationRisk = await this.calculateConcentrationRisk(portfolio);

    return {
      var95,
      var99,
      expectedShortfall,
      maxDrawdown,
      beta,
      concentrationRisk,
      liquidityRisk: await this.calculateLiquidityRisk(portfolio),
      correlationRisk: await this.calculateCorrelationRisk(portfolio)
    };
  }

  async getYieldAnalytics(portfolioId: string): Promise<YieldMetrics> {
    const positions = await this.defiPositionRepository.find({
      where: { portfolioId, isActive: true }
    });

    let totalYield = 0;
    let weightedApy = 0;
    let farmingRewards = 0;
    let stakingRewards = 0;

    for (const position of positions) {
      if (position.positionData.yield) {
        totalYield += position.positionData.yield.amount || 0;
        weightedApy += (position.positionData.apy || 0) * (position.value / 100);
      }

      if (position.positionData.rewards) {
        farmingRewards += position.positionData.rewards.farming || 0;
        stakingRewards += position.positionData.rewards.staking || 0;
      }
    }

    return {
      totalYield,
      weightedApy,
      farmingRewards,
      stakingRewards,
      annualizedReturn: weightedApy,
      yieldHistory: await this.getYieldHistory(portfolioId)
    };
  }

  async getLiquidityAnalytics(protocol: string): Promise<LiquidityMetrics> {
    // Get liquidity pools for protocol
    const pools = await this.defiPositionRepository
      .createQueryBuilder('position')
      .where('position.protocol = :protocol', { protocol })
      .andWhere('position.isActive = true')
      .groupBy('position.poolAddress')
      .select(['position.poolAddress', 'SUM(position.value) as totalLiquidity'])
      .getRawMany();

    const totalLiquidity = pools.reduce((sum, pool) => sum + parseFloat(pool.totalLiquidity), 0);
    const averageDepth = totalLiquidity / Math.max(pools.length, 1);

    // Calculate liquidity utilization
    const utilizationRate = await this.calculateLiquidityUtilization(protocol);

    // Calculate impermanent loss risk
    const impermanentLossRisk = await this.calculateImpermanentLossRisk(protocol);

    return {
      totalLiquidity,
      pools: pools.length,
      averageDepth,
      utilizationRate,
      impermanentLossRisk,
      liquidityMigration: await this.calculateLiquidityMigration(protocol)
    };
  }

  async getSocialSentimentAnalytics(protocol: string): Promise<SocialSentimentMetrics> {
    const cacheKey = `analytics:sentiment:${protocol}`;
    
    // Check cache
    if (this.configService.get('analytics.cacheEnabled')) {
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    // This would integrate with the social data service
    const sentimentData = await this.getSentimentData(protocol);
    
    // Cache for 15 minutes
    await this.redis.setex(cacheKey, 900, JSON.stringify(sentimentData));

    return sentimentData;
  }

  async getTrendAnalysis(timeframe: '1h' | '1d' | '7d' | '30d'): Promise<TrendAnalysis> {
    const timeRange = this.getTimeRangeFromTimeframe(timeframe);
    
    // Get trending protocols
    const trendingProtocols = await this.getTrendingProtocols(timeRange);
    
    // Get market trends
    const marketTrends = await this.getMarketTrends(timeRange);
    
    // Get yield trends
    const yieldTrends = await this.getYieldTrends(timeRange);

    return {
      timestamp: new Date(),
      timeframe,
      trendingProtocols,
      marketTrends,
      yieldTrends,
      volatilityTrends: await this.getVolatilityTrends(timeRange)
    };
  }

  async getAlertAnalytics(userId: string): Promise<AlertMetrics> {
    const alerts = await this.alertRepository.find({
      where: { userId },
      order: { createdAt: 'DESC' }
    });

    const totalAlerts = alerts.length;
    const triggeredAlerts = alerts.filter(alert => alert.isTriggered).length;
    const activeAlerts = alerts.filter(alert => alert.isActive).length;

    // Get alert types breakdown
    const alertTypes = alerts.reduce((acc, alert) => {
      acc[alert.alertType] = (acc[alert.alertType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Calculate alert frequency
    const alertFrequency = await this.calculateAlertFrequency(userId);

    return {
      totalAlerts,
      triggeredAlerts,
      activeAlerts,
      alertTypes,
      alertFrequency,
      recentAlerts: alerts.slice(0, 10)
    };
  }

  // Real-time analytics updates
  @Cron('*/1 * * * *') // Every minute
  async updateRealTimeAnalytics() {
    if (!this.configService.get('analytics.enableRealTimeUpdates')) {
      return;
    }

    try {
      // Update portfolio analytics
      await this.updatePortfolioAnalytics();
      
      // Update protocol metrics
      await this.updateProtocolMetrics();
      
      // Update market metrics
      await this.updateMarketMetrics();

      this.logger.log('Real-time analytics updated successfully');
    } catch (error) {
      this.logger.error('Failed to update real-time analytics', error);
    }
  }

  // Utility functions
  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = math.mean(returns);
    const stdDev = math.std(returns);
    
    return stdDev > 0 ? avgReturn / stdDev : 0;
  }

  private calculateMaxDrawdown(values: number[]): number {
    if (values.length === 0) return 0;
    
    let maxDrawdown = 0;
    let peak = values[0];
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
      }
      
      const drawdown = (peak - value) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown;
  }

  private calculateVolatility(returns: number[]): number {
    if (returns.length === 0) return 0;
    return math.std(returns);
  }

  private calculateAlphaBeta(portfolioReturns: number[], marketReturns: number[]): { alpha: number; beta: number } {
    if (portfolioReturns.length !== marketReturns.length || portfolioReturns.length === 0) {
      return { alpha: 0, beta: 0 };
    }

    // Calculate correlation and covariance
    const correlation = this.calculateCorrelation(portfolioReturns, marketReturns);
    const portfolioVolatility = math.std(portfolioReturns);
    const marketVolatility = math.std(marketReturns);
    
    // Beta = correlation * (portfolio volatility / market volatility)
    const beta = marketVolatility > 0 ? correlation * (portfolioVolatility / marketVolatility) : 0;
    
    // Alpha = portfolio mean return - (beta * market mean return)
    const alpha = math.mean(portfolioReturns) - (beta * math.mean(marketReturns));
    
    return { alpha, beta };
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const xMean = math.mean(x);
    const yMean = math.mean(y);
    
    let numerator = 0;
    let xSumSquares = 0;
    let ySumSquares = 0;
    
    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - xMean;
      const yDiff = y[i] - yMean;
      
      numerator += xDiff * yDiff;
      xSumSquares += xDiff * xDiff;
      ySumSquares += yDiff * yDiff;
    }
    
    const denominator = Math.sqrt(xSumSquares * ySumSquares);
    return denominator > 0 ? numerator / denominator : 0;
  }

  private calculateRiskMetrics(returns: number[]) {
    if (returns.length === 0) {
      return {
        var95: 0,
        var99: 0,
        expectedShortfall: 0,
        maxDrawdown: 0,
        sharpeRatio: 0
      };
    }

    const sortedReturns = returns.sort((a, b) => a - b);
    const var95Index = Math.floor(returns.length * 0.05);
    const var99Index = Math.floor(returns.length * 0.01);
    
    const var95 = sortedReturns[var95Index] || 0;
    const var99 = sortedReturns[var99Index] || 0;
    
    // Expected Shortfall (average of returns worse than VaR)
    const tailReturns = sortedReturns.slice(0, var95Index);
    const expectedShortfall = tailReturns.length > 0 ? math.mean(tailReturns) : 0;
    
    return {
      var95: Math.abs(var95),
      var99: Math.abs(var99),
      expectedShortfall: Math.abs(expectedShortfall),
      maxDrawdown: this.calculateMaxDrawdown(returns),
      sharpeRatio: this.calculateSharpeRatio(returns)
    };
  }

  private async calculatePortfolioValue(portfolio: Portfolio): Promise<number> {
    // This would integrate with price feeds to get current values
    let totalValue = 0;
    
    for (const position of portfolio.positions || []) {
      // Get current price and calculate position value
      const positionValue = await this.getPositionValue(position);
      totalValue += positionValue;
    }
    
    return totalValue;
  }

  private async getPositionValue(position: any): Promise<number> {
    // Implementation would fetch current prices and calculate position value
    return position.value || 0;
  }

  private getTimeRangeFromTimeframe(timeframe: string): TimeRange {
    const now = new Date();
    
    switch (timeframe) {
      case '1h':
        return { from: subDays(now, 0), to: now };
      case '1d':
        return { from: subDays(now, 1), to: now };
      case '7d':
        return { from: subDays(now, 7), to: now };
      case '30d':
        return { from: subDays(now, 30), to: now };
      default:
        return { from: subDays(now, 7), to: now };
    }
  }

  // Additional utility methods would be implemented here...
}