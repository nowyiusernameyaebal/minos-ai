import { expect } from 'chai';
import nock from 'nock';
import { MinosClient } from '../src/client';
import { AgentClient } from '../src/agent';
import { Agent, AgentPerformance, AgentTrade, BacktestResult } from '../src/types';

describe('AgentClient', () => {
  // Constants
  const API_URL = 'https://api.minos.ai';
  const API_KEY = 'test-api-key';
  const API_SECRET = 'test-api-secret';
  
  // Test user data
  const testUser = {
    id: '9f5a9e2f-8b2a-4f9a-b8e7-c9c8b2a2a4a8',
    email: 'test@example.com',
    name: 'Test User'
  };
  
  // Test tokens
  const testTokens = {
    accessToken: 'test-access-token',
    refreshToken: 'test-refresh-token',
    expiresIn: 3600
  };

  // Test vault data
  const testVault = {
    id: 'v-123456',
    name: 'Test Vault',
    userId: testUser.id
  };

  // Test agent data
  const testAgent: Agent = {
    id: 'a-123456',
    name: 'Test AI Agent',
    description: 'AI agent for testing',
    vaultId: testVault.id,
    model: 'Ariadne-v1.0',
    strategyType: 'MOMENTUM',
    riskLevel: 3, // Medium risk
    maxPositionSize: '10000',
    maxDailyTrades: 10,
    status: 'ACTIVE',
    constraints: {
      maxPositionSize: '10000',
      maxDailyTrades: 10,
      allowedMarkets: ['CRYPTO', 'FOREX'],
      blacklistedTokens: []
    },
    parameters: {
      indicators: ['sma', 'ema', 'rsi', 'macd'],
      timeframes: ['1m', '5m', '15m', '1h'],
      entryThreshold: 0.7,
      exitThreshold: 0.3
    },
    createdAt: '2025-01-01T00:00:00Z',
    updatedAt: '2025-01-01T00:00:00Z'
  };
  
  // Test performance data
  const testPerformance: AgentPerformance = {
    agentId: testAgent.id,
    totalPnl: '2500',
    winRate: 0.68,
    totalTrades: 50,
    successfulTrades: 34,
    failedTrades: 16,
    sharpeRatio: 1.9,
    maxDrawdown: -0.1,
    averageTradeSize: '800',
    timestamp: '2025-01-03T00:00:00Z'
  };
  
  // Test trade data
  const testTrade: AgentTrade = {
    id: 'trade-123456',
    agentId: testAgent.id,
    vaultId: testVault.id,
    asset: 'BTC-USD',
    direction: 'BUY',
    amount: '0.5',
    price: '45000',
    status: 'EXECUTED',
    pnl: '500',
    executedAt: '2025-01-02T00:00:00Z'
  };

  // Test backtest data
  const testBacktest: BacktestResult = {
    id: 'b-123456',
    agentId: testAgent.id,
    status: 'COMPLETED',
    startDate: '2024-01-01T00:00:00Z',
    endDate: '2024-12-31T23:59:59Z',
    assets: ['BTC-USD', 'ETH-USD'],
    initialCapital: '10000',
    finalCapital: '12500',
    totalReturn: 0.25,
    annualizedReturn: 0.32,
    maxDrawdown: -0.15,
    sharpeRatio: 1.5,
    totalTrades: 45,
    winRate: 0.6,
    averageWin: 250,
    averageLoss: -150,
    profitFactor: 1.67,
    createdAt: '2025-01-04T00:00:00Z',
    completedAt: '2025-01-04T00:05:00Z',
    results: {
      equity_curve: [10000, 10100, 10050, 10200, 10500, 12500],
      trades: [
        {
          asset: 'BTC-USD',
          direction: 'BUY',
          entryPrice: '40000',
          exitPrice: '42000',
          pnl: '200',
          timestamp: '2024-01-15T00:00:00Z'
        },
        {
          asset: 'ETH-USD',
          direction: 'BUY',
          entryPrice: '2000',
          exitPrice: '2200',
          pnl: '100',
          timestamp: '2024-02-10T00:00:00Z'
        }
      ]
    }
  };

  let minosClient: MinosClient;
  let agentClient: AgentClient;

  beforeEach(async () => {
    // Create a new client for each test
    minosClient = new MinosClient({
      apiUrl: API_URL,
      apiKey: API_KEY,
      apiSecret: API_SECRET
    });
    
    // Authenticate
    nock(API_URL)
      .post('/auth/api-key')
      .reply(200, {
        user: testUser,
        tokens: testTokens
      });

    await minosClient.authenticate();
    
    // Create agent client
    agentClient = new AgentClient(minosClient);
    
    // Reset any previous nock interceptors for API endpoints
    nock.cleanAll();
  });

  afterEach(() => {
    // Ensure all nock interceptors have been used
    expect(nock.isDone()).to.be.true;
  });

  describe('List Agents', () => {
    it('should list user agents', async () => {
      nock(API_URL)
        .get('/agents')
        .reply(200, [testAgent]);

      const agents = await agentClient.listAgents();
      
      expect(agents).to.be.an('array');
      expect(agents.length).to.equal(1);
      expect(agents[0].id).to.equal(testAgent.id);
    });

    it('should support pagination', async () => {
      nock(API_URL)
        .get('/agents?page=2&limit=10')
        .reply(200, [testAgent]);

      const agents = await agentClient.listAgents({ page: 2, limit: 10 });
      
      expect(agents).to.be.an('array');
      expect(agents.length).to.equal(1);
    });

    it('should filter agents by vault ID', async () => {
      nock(API_URL)
        .get(`/agents?vaultId=${testVault.id}`)
        .reply(200, [testAgent]);

      const agents = await agentClient.listAgents({ vaultId: testVault.id });
      
      expect(agents).to.be.an('array');
      expect(agents.length).to.equal(1);
      expect(agents[0].vaultId).to.equal(testVault.id);
    });

    it('should filter agents by status', async () => {
      nock(API_URL)
        .get('/agents?status=ACTIVE')
        .reply(200, [testAgent]);

      const agents = await agentClient.listAgents({ status: 'ACTIVE' });
      
      expect(agents).to.be.an('array');
      expect(agents.length).to.equal(1);
      expect(agents[0].status).to.equal('ACTIVE');
    });
  });

  describe('Get Agent', () => {
    it('should get an agent by ID', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}`)
        .reply(200, testAgent);

      const agent = await agentClient.getAgent(testAgent.id);
      
      expect(agent).to.deep.equal(testAgent);
    });

    it('should throw an error if agent not found', async () => {
      const nonExistentId = 'a-nonexistent';
      
      nock(API_URL)
        .get(`/agents/${nonExistentId}`)
        .reply(404, {
          message: 'Agent not found'
        });

      try {
        await agentClient.getAgent(nonExistentId);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Agent not found');
        expect(error.status).to.equal(404);
      }
    });
  });

  describe('Create Agent', () => {
    it('should create a new agent', async () => {
      const agentData = {
        name: 'New AI Agent',
        description: 'A new AI agent for testing',
        vaultId: testVault.id,
        model: 'Ariadne-v1.0',
        strategyType: 'MEAN_REVERSION',
        riskLevel: 2,
        maxPositionSize: '5000',
        maxDailyTrades: 5
      };
      
      const newAgent = {
        ...testAgent,
        id: 'a-new123',
        name: agentData.name,
        description: agentData.description,
        strategyType: agentData.strategyType,
        riskLevel: agentData.riskLevel,
        maxPositionSize: agentData.maxPositionSize,
        maxDailyTrades: agentData.maxDailyTrades
      };
      
      nock(API_URL)
        .post('/agents', agentData)
        .reply(201, newAgent);

      const agent = await agentClient.createAgent(agentData);
      
      expect(agent).to.deep.equal(newAgent);
    });

    it('should validate required fields', async () => {
      const invalidData = {
        // Missing name and vaultId
        description: 'Invalid agent'
      };
      
      nock(API_URL)
        .post('/agents', invalidData)
        .reply(400, {
          message: 'Validation failed',
          errors: ['name is required', 'vaultId is required']
        });

      try {
        await agentClient.createAgent(invalidData);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Validation failed');
        expect(error.status).to.equal(400);
        expect(error.errors).to.deep.equal(['name is required', 'vaultId is required']);
      }
    });
  });

  describe('Update Agent', () => {
    it('should update an agent', async () => {
      const updateData = {
        name: 'Updated AI Agent',
        description: 'Updated description',
        riskLevel: 2,
        maxDailyTrades: 15
      };
      
      const updatedAgent = {
        ...testAgent,
        name: updateData.name,
        description: updateData.description,
        riskLevel: updateData.riskLevel,
        maxDailyTrades: updateData.maxDailyTrades,
        updatedAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .put(`/agents/${testAgent.id}`, updateData)
        .reply(200, updatedAgent);

      const agent = await agentClient.updateAgent(testAgent.id, updateData);
      
      expect(agent).to.deep.equal(updatedAgent);
    });
  });

  describe('Delete Agent', () => {
    it('should delete an agent', async () => {
      nock(API_URL)
        .delete(`/agents/${testAgent.id}`)
        .reply(200, {
          success: true,
          message: 'Agent deleted successfully'
        });

      await agentClient.deleteAgent(testAgent.id);
      // If no error is thrown, the test passes
    });

    it('should throw an error if agent cannot be deleted', async () => {
      nock(API_URL)
        .delete(`/agents/${testAgent.id}`)
        .reply(400, {
          message: 'Cannot delete active agent'
        });

      try {
        await agentClient.deleteAgent(testAgent.id);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Cannot delete active agent');
        expect(error.status).to.equal(400);
      }
    });
  });

  describe('Activate and Deactivate Agent', () => {
    it('should activate an agent', async () => {
      const inactiveAgent = {
        ...testAgent,
        status: 'INACTIVE'
      };
      
      const activatedAgent = {
        ...testAgent,
        status: 'ACTIVE',
        updatedAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .post(`/agents/${inactiveAgent.id}/activate`)
        .reply(200, activatedAgent);

      const agent = await agentClient.activateAgent(inactiveAgent.id);
      
      expect(agent.status).to.equal('ACTIVE');
    });

    it('should deactivate an agent', async () => {
      const deactivatedAgent = {
        ...testAgent,
        status: 'INACTIVE',
        updatedAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .post(`/agents/${testAgent.id}/deactivate`)
        .reply(200, deactivatedAgent);

      const agent = await agentClient.deactivateAgent(testAgent.id);
      
      expect(agent.status).to.equal('INACTIVE');
    });
  });

  describe('Agent Parameters', () => {
    it('should update agent parameters', async () => {
      const parameters = {
        indicators: ['bollinger', 'stochastic'],
        timeframes: ['4h', '1d'],
        entryThreshold: 0.8,
        exitThreshold: 0.2
      };
      
      const updatedAgent = {
        ...testAgent,
        parameters,
        updatedAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .post(`/agents/${testAgent.id}/parameters`, { parameters })
        .reply(200, updatedAgent);

      const agent = await agentClient.updateParameters(testAgent.id, parameters);
      
      expect(agent.parameters).to.deep.equal(parameters);
    });
  });

  describe('Agent Performance', () => {
    it('should get agent performance', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/performance`)
        .reply(200, testPerformance);

      const performance = await agentClient.getPerformance(testAgent.id);
      
      expect(performance).to.deep.equal(testPerformance);
    });
  });

  describe('Agent Trades', () => {
    it('should list agent trades', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/trades`)
        .reply(200, [testTrade]);

      const trades = await agentClient.listTrades(testAgent.id);
      
      expect(trades).to.be.an('array');
      expect(trades.length).to.equal(1);
      expect(trades[0].id).to.equal(testTrade.id);
    });

    it('should filter trades by date range', async () => {
      const startDate = '2025-01-01T00:00:00Z';
      const endDate = '2025-01-03T00:00:00Z';
      
      nock(API_URL)
        .get(`/agents/${testAgent.id}/trades?startDate=${encodeURIComponent(startDate)}&endDate=${encodeURIComponent(endDate)}`)
        .reply(200, [testTrade]);

      const trades = await agentClient.listTrades(testAgent.id, { startDate, endDate });
      
      expect(trades).to.be.an('array');
      expect(trades.length).to.equal(1);
    });

    it('should filter trades by asset', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/trades?asset=BTC-USD`)
        .reply(200, [testTrade]);

      const trades = await agentClient.listTrades(testAgent.id, { asset: 'BTC-USD' });
      
      expect(trades).to.be.an('array');
      expect(trades.length).to.equal(1);
      expect(trades[0].asset).to.equal('BTC-USD');
    });
  });

  describe('Backtesting', () => {
    it('should start a backtest', async () => {
      const backtestParams = {
        startDate: '2024-01-01T00:00:00Z',
        endDate: '2024-12-31T23:59:59Z',
        assets: ['BTC-USD', 'ETH-USD'],
        initialCapital: '10000'
      };
      
      const pendingBacktest = {
        id: 'b-new123',
        agentId: testAgent.id,
        status: 'PENDING',
        ...backtestParams,
        createdAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .post(`/agents/${testAgent.id}/backtest`, backtestParams)
        .reply(202, pendingBacktest);

      const backtest = await agentClient.runBacktest(testAgent.id, backtestParams);
      
      expect(backtest.id).to.equal(pendingBacktest.id);
      expect(backtest.status).to.equal('PENDING');
      expect(backtest.agentId).to.equal(testAgent.id);
    });

    it('should get backtest result', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/backtest/${testBacktest.id}`)
        .reply(200, testBacktest);

      const backtest = await agentClient.getBacktestResult(testAgent.id, testBacktest.id);
      
      expect(backtest).to.deep.equal(testBacktest);
    });

    it('should list backtests', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/backtests`)
        .reply(200, [testBacktest]);

      const backtests = await agentClient.listBacktests(testAgent.id);
      
      expect(backtests).to.be.an('array');
      expect(backtests.length).to.equal(1);
      expect(backtests[0].id).to.equal(testBacktest.id);
    });

    it('should filter backtests by status', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/backtests?status=COMPLETED`)
        .reply(200, [testBacktest]);

      const backtests = await agentClient.listBacktests(testAgent.id, { status: 'COMPLETED' });
      
      expect(backtests).to.be.an('array');
      expect(backtests.length).to.equal(1);
      expect(backtests[0].status).to.equal('COMPLETED');
    });
  });

  describe('Available Models and Strategies', () => {
    it('should get available AI models', async () => {
      const models = [
        {
          id: 'ariadne-v1.0',
          name: 'Ariadne-v1.0',
          description: 'Advanced momentum trading model',
          capabilities: ['CRYPTO', 'FOREX', 'STOCKS'],
          created: '2024-12-01T00:00:00Z'
        },
        {
          id: 'androgeus-v1.0',
          name: 'Androgeus-v1.0',
          description: 'Technical analysis focused model',
          capabilities: ['CRYPTO', 'FOREX'],
          created: '2025-01-01T00:00:00Z'
        }
      ];
      
      nock(API_URL)
        .get('/agents/models')
        .reply(200, models);

      const availableModels = await agentClient.getAvailableModels();
      
      expect(availableModels).to.be.an('array');
      expect(availableModels.length).to.equal(2);
      expect(availableModels[0].id).to.equal('ariadne-v1.0');
      expect(availableModels[1].id).to.equal('androgeus-v1.0');
    });

    it('should get available strategy types', async () => {
      const strategies = [
        {
          id: 'MOMENTUM',
          name: 'Momentum Trading',
          description: 'Follows market trends and momentum indicators',
          riskLevels: [1, 2, 3, 4, 5]
        },
        {
          id: 'MEAN_REVERSION',
          name: 'Mean Reversion',
          description: 'Identifies and trades price reversals to the mean',
          riskLevels: [1, 2, 3, 4, 5]
        }
      ];
      
      nock(API_URL)
        .get('/agents/strategies')
        .reply(200, strategies);

      const availableStrategies = await agentClient.getAvailableStrategies();
      
      expect(availableStrategies).to.be.an('array');
      expect(availableStrategies.length).to.equal(2);
      expect(availableStrategies[0].id).to.equal('MOMENTUM');
      expect(availableStrategies[1].id).to.equal('MEAN_REVERSION');
    });
  });

  describe('Analytics and Reporting', () => {
    it('should get agent analytics', async () => {
      const analytics = {
        performance: {
          daily: [0.01, 0.02, -0.005, 0.015],
          weekly: [0.03, 0.02],
          monthly: [0.05]
        },
        tradingPatterns: {
          byAsset: [
            { asset: 'BTC-USD', trades: 30, winRate: 0.7 },
            { asset: 'ETH-USD', trades: 15, winRate: 0.6 }
          ],
          byTimeOfDay: [
            { hour: 0, trades: 2, winRate: 0.5 },
            { hour: 1, trades: 3, winRate: 0.67 },
            // ... other hours
            { hour: 23, trades: 1, winRate: 1.0 }
          ],
          byDayOfWeek: [
            { day: 'Monday', trades: 10, winRate: 0.6 },
            // ... other days
            { day: 'Sunday', trades: 5, winRate: 0.8 }
          ]
        },
        riskAnalysis: {
          drawdowns: [
            { start: '2025-01-01T00:00:00Z', end: '2025-01-03T00:00:00Z', depth: -0.05, recovery: '2025-01-05T00:00:00Z' }
          ],
          volatility: 0.02,
          profitLossRatio: 1.5,
          sharpeRatio: 1.9,
          sortinoRatio: 2.1
        }
      };
      
      nock(API_URL)
        .get(`/agents/${testAgent.id}/analytics`)
        .reply(200, analytics);

      const agentAnalytics = await agentClient.getAnalytics(testAgent.id);
      
      expect(agentAnalytics).to.deep.equal(analytics);
    });

    it('should export agent data', async () => {
      nock(API_URL)
        .get(`/agents/${testAgent.id}/export?format=json`)
        .reply(200, {
          agent: testAgent,
          performance: testPerformance,
          trades: [testTrade],
          backtests: [testBacktest],
          exportedAt: '2025-01-05T00:00:00Z'
        });

      const exportData = await agentClient.exportData(testAgent.id, 'json');
      
      expect(exportData).to.have.property('agent');
      expect(exportData).to.have.property('performance');
      expect(exportData).to.have.property('trades');
      expect(exportData).to.have.property('backtests');
      expect(exportData).to.have.property('exportedAt');
    });
  });
});