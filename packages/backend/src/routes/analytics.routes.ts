/**
 * Analytics Routes for Minos-AI DeFi Platform
 * REST API endpoints for portfolio, protocol, and market analytics
 */

import { Router } from 'express';
import { AnalyticsController } from '../controllers/analytics.controller';
import { authenticateToken } from '../middleware/auth.middleware';
import { rateLimiter } from '../middleware/rate-limiter.middleware';
import { validateDto } from '../utils/validation';
import { 
  GetPortfolioAnalyticsDto,
  GetProtocolAnalyticsDto,
  GetMarketAnalyticsDto,
  GetRiskAnalyticsDto,
  GetYieldAnalyticsDto,
  GetTrendAnalysisDto,
  GetAlertAnalyticsDto,
  GetTimeRangeDto,
  QueryFilterDto,
  ExportOptionsDto,
  CompareEntitiesDto,
  CustomQueryDto,
  PaginationDto
} from '../dto/analytics.dto';
import { cache } from '../middleware/cache.middleware';
import { roleGuard } from '../middleware/role-guard.middleware';
import { validateTimeRange } from '../middleware/time-range.middleware';
import { sanitizeInput } from '../middleware/sanitize.middleware';

const router = Router();
const analyticsController = new AnalyticsController();

// Apply authentication and sanitization to all routes
router.use(authenticateToken);
router.use(sanitizeInput);

// Portfolio Analytics Routes
router.get(
  '/portfolios/:portfolioId',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }), // 5 minute cache
  validateDto(GetPortfolioAnalyticsDto),
  validateTimeRange,
  analyticsController.getPortfolioAnalytics
);

router.get(
  '/portfolios/:portfolioId/performance',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getPortfolioPerformance
);

router.get(
  '/portfolios/:portfolioId/risk',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }), // 10 minute cache
  validateDto(GetRiskAnalyticsDto),
  analyticsController.getRiskAnalytics
);

router.get(
  '/portfolios/:portfolioId/yield',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetYieldAnalyticsDto),
  analyticsController.getYieldAnalytics
);

router.get(
  '/portfolios/:portfolioId/asset-allocation',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  analyticsController.getAssetAllocation
);

router.get(
  '/portfolios/:portfolioId/historical',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getHistoricalData
);

router.get(
  '/portfolios/:portfolioId/sharpe-ratio',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getPortfolioSharpeRatio
);

router.get(
  '/portfolios/:portfolioId/value-at-risk',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getPortfolioVaR
);

router.get(
  '/portfolios/:portfolioId/correlation-matrix',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 900 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getPortfolioCorrelationMatrix
);

router.get(
  '/portfolios/:portfolioId/transaction-analysis',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getTransactionAnalysis
);

router.get(
  '/portfolios/:portfolioId/profit-loss',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProfitLossAnalysis
);

router.get(
  '/portfolios/:portfolioId/fee-analysis',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getFeeAnalysis
);

router.get(
  '/portfolios/:portfolioId/rebalancing-suggestions',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 300 }),
  analyticsController.getRebalancingSuggestions
);

// Multi-portfolio Analytics
router.get(
  '/portfolios/compare',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  validateDto(CompareEntitiesDto),
  analyticsController.comparePortfolios
);

router.get(
  '/portfolios/benchmarks',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getPortfolioBenchmarks
);

router.get(
  '/portfolios/rankings',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(PaginationDto),
  analyticsController.getPortfolioRankings
);

// Protocol Analytics Routes
router.get(
  '/protocols/:protocol',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetProtocolAnalyticsDto),
  analyticsController.getProtocolAnalytics
);

router.get(
  '/protocols/:protocol/metrics',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolMetrics
);

router.get(
  '/protocols/:protocol/tvl',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolTvl
);

router.get(
  '/protocols/:protocol/users',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolUsers
);

router.get(
  '/protocols/:protocol/liquidity',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolLiquidity
);

router.get(
  '/protocols/:protocol/volume',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolVolume
);

router.get(
  '/protocols/:protocol/yield',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolYield
);

router.get(
  '/protocols/:protocol/fees',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getProtocolFees
);

router.get(
  '/protocols/:protocol/pools',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  analyticsController.getProtocolPools
);

router.get(
  '/protocols/:protocol/governance',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  analyticsController.getProtocolGovernance
);

router.get(
  '/protocols/:protocol/risk-metrics',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  analyticsController.getProtocolRiskMetrics
);

router.get(
  '/protocols/:protocol/security-score',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  analyticsController.getProtocolSecurityScore
);

router.get(
  '/protocols/:protocol/ecosystem',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  analyticsController.getProtocolEcosystem
);

// Multi-protocol Analytics
router.get(
  '/protocols/compare',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(CompareEntitiesDto),
  analyticsController.compareProtocols
);

router.get(
  '/protocols/rankings',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(PaginationDto),
  analyticsController.getProtocolRankings
);

router.get(
  '/protocols/market-share',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  analyticsController.getProtocolMarketShare
);

// Market Analytics Routes
router.get(
  '/market',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 300 }),
  validateDto(GetMarketAnalyticsDto),
  analyticsController.getMarketAnalytics
);

router.get(
  '/market/overview',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 300 }),
  analyticsController.getMarketOverview
);

router.get(
  '/market/trends',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 300 }),
  validateDto(GetTrendAnalysisDto),
  analyticsController.getTrendAnalysis
);

router.get(
  '/market/sentiment',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 900 }), // 15 minute cache
  analyticsController.getMarketSentiment
);

router.get(
  '/market/dominance',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 300 }),
  analyticsController.getMarketDominance
);

router.get(
  '/market/volatility',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getMarketVolatility
);

router.get(
  '/market/fear-greed-index',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 600 }),
  analyticsController.getFearGreedIndex
);

router.get(
  '/market/sectors',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 600 }),
  analyticsController.getMarketSectors
);

router.get(
  '/market/news-sentiment',
  rateLimiter({ windowMs: 60000, max: 60 }),
  cache({ ttl: 900 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getNewsSentiment
);

router.get(
  '/market/whale-movements',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getWhaleMovements
);

router.get(
  '/market/arbitrage-opportunities',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 60 }), // 1 minute cache
  analyticsController.getArbitrageOpportunities
);

// Token Analytics Routes
router.get(
  '/tokens/:tokenAddress/analytics',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getTokenAnalytics
);

router.get(
  '/tokens/:tokenAddress/price-history',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getTokenPriceHistory
);

router.get(
  '/tokens/:tokenAddress/holders',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  validateDto(PaginationDto),
  analyticsController.getTokenHolders
);

router.get(
  '/tokens/:tokenAddress/distribution',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 600 }),
  analyticsController.getTokenDistribution
);

router.get(
  '/tokens/:tokenAddress/pair-analysis',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getTokenPairAnalysis
);

// Social Analytics Routes
router.get(
  '/social/sentiment/:protocol',
  rateLimiter({ windowMs: 60000, max: 40 }),
  cache({ ttl: 900 }), // 15 minute cache
  validateDto(GetTimeRangeDto),
  analyticsController.getSocialSentiment
);

router.get(
  '/social/trends',
  rateLimiter({ windowMs: 60000, max: 40 }),
  cache({ ttl: 900 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getSocialTrends
);

router.get(
  '/social/influencers/:protocol',
  rateLimiter({ windowMs: 60000, max: 40 }),
  cache({ ttl: 1800 }),
  validateDto(PaginationDto),
  analyticsController.getInfluencerAnalysis
);

router.get(
  '/social/mentions/:protocol',
  rateLimiter({ windowMs: 60000, max: 40 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getSocialMentions
);

router.get(
  '/social/correlation/:protocol',
  rateLimiter({ windowMs: 60000, max: 40 }),
  cache({ ttl: 900 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getSocialPriceCorrelation
);

// Advanced Analytics Routes
router.get(
  '/forecasting/:type',
  rateLimiter({ windowMs: 60000, max: 10 }),
  cache({ ttl: 1800 }), // 30 minute cache
  analyticsController.getForecast
);

router.get(
  '/correlations',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 900 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getCorrelationMatrix
);

router.get(
  '/sectors/performance',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getSectorPerformance
);

router.get(
  '/sectors/rotation',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getSectorRotation
);

router.get(
  '/risk/heatmap',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  analyticsController.getRiskHeatmap
);

router.get(
  '/yield/strategies',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(QueryFilterDto),
  analyticsController.getYieldStrategies
);

router.get(
  '/yield/leaderboard',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(PaginationDto),
  analyticsController.getYieldLeaderboard
);

router.get(
  '/yield/farmer-analysis',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  roleGuard(['premium', 'admin']),
  analyticsController.getYieldFarmerAnalysis
);

// Alert Analytics Routes
router.get(
  '/alerts/metrics',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetAlertAnalyticsDto),
  analyticsController.getAlertAnalytics
);

router.get(
  '/alerts/performance',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getAlertPerformance
);

router.get(
  '/alerts/effectiveness',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  analyticsController.getAlertEffectiveness
);

router.get(
  '/alerts/summary',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 300 }),
  analyticsController.getAlertSummary
);

// Recommendation Engine Routes
router.get(
  '/recommendations/portfolios/:portfolioId',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  analyticsController.getPortfolioRecommendations
);

router.get(
  '/recommendations/protocols',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  validateDto(QueryFilterDto),
  analyticsController.getProtocolRecommendations
);

router.get(
  '/recommendations/yield-farming',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  validateDto(QueryFilterDto),
  analyticsController.getYieldFarmingRecommendations
);

// Export Analytics Routes
router.get(
  '/export/portfolio/:portfolioId',
  rateLimiter({ windowMs: 60000, max: 10 }),
  validateDto(ExportOptionsDto),
  analyticsController.exportPortfolioData
);

router.get(
  '/export/market/:format',
  rateLimiter({ windowMs: 60000, max: 10 }),
  validateDto(ExportOptionsDto),
  analyticsController.exportMarketData
);

router.get(
  '/export/protocols/:protocol',
  rateLimiter({ windowMs: 60000, max: 10 }),
  validateDto(ExportOptionsDto),
  analyticsController.exportProtocolData
);

router.get(
  '/export/custom-report',
  rateLimiter({ windowMs: 60000, max: 5 }),
  validateDto(CustomQueryDto),
  roleGuard(['premium', 'admin']),
  analyticsController.exportCustomReport
);

// Real-time Analytics Routes
router.get(
  '/realtime/prices',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getRealTimePrices
);

router.get(
  '/realtime/portfolio/:portfolioId',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getRealTimePortfolioData
);

router.get(
  '/realtime/market-events',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getRealTimeMarketEvents
);

router.get(
  '/realtime/liquidations',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getRealTimeLiquidations
);

router.get(
  '/realtime/arbitrage',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getRealTimeArbitrage
);

// Performance Metrics Routes
router.get(
  '/performance/benchmarks',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(GetTimeRangeDto),
  analyticsController.getBenchmarks
);

router.get(
  '/performance/rankings',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  validateDto(PaginationDto),
  analyticsController.getPerformanceRankings
);

router.get(
  '/performance/attribution',
  rateLimiter({ windowMs: 60000, max: 30 }),
  cache({ ttl: 600 }),
  roleGuard(['premium', 'admin']),
  analyticsController.getPerformanceAttribution
);

router.get(
  '/performance/stress-test',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 1800 }),
  roleGuard(['premium', 'admin']),
  analyticsController.getStressTestResults
);

// Custom Analytics Routes
router.post(
  '/custom/query',
  rateLimiter({ windowMs: 60000, max: 5 }),
  validateDto(CustomQueryDto),
  roleGuard(['premium', 'admin']),
  analyticsController.executeCustomQuery
);

router.post(
  '/custom/backtest',
  rateLimiter({ windowMs: 60000, max: 3 }),
  roleGuard(['premium', 'admin']),
  analyticsController.runBacktest
);

router.get(
  '/dashboards/:dashboardId',
  rateLimiter({ windowMs: 60000, max: 50 }),
  cache({ ttl: 300 }),
  analyticsController.getDashboardAnalytics
);

router.post(
  '/dashboards/:dashboardId/refresh',
  rateLimiter({ windowMs: 60000, max: 10 }),
  analyticsController.refreshDashboard
);

// Admin Analytics Routes
router.get(
  '/admin/system-metrics',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 60 }),
  roleGuard(['admin']),
  analyticsController.getSystemMetrics
);

router.get(
  '/admin/usage-statistics',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 300 }),
  roleGuard(['admin']),
  analyticsController.getUsageStatistics
);

router.get(
  '/admin/error-analytics',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 300 }),
  roleGuard(['admin']),
  analyticsController.getErrorAnalytics
);

router.get(
  '/admin/performance-bottlenecks',
  rateLimiter({ windowMs: 60000, max: 20 }),
  cache({ ttl: 600 }),
  roleGuard(['admin']),
  analyticsController.getPerformanceBottlenecks
);

// Health Check Routes
router.get(
  '/health',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getHealthStatus
);

router.get(
  '/status',
  rateLimiter({ windowMs: 60000, max: 100 }),
  analyticsController.getSystemStatus
);

export default router;