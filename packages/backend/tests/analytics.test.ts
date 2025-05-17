import request from 'supertest';
import { expect } from 'chai';
import { app } from '../src/app';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';
import { hashPassword } from '../src/utils/auth';

const prisma = new PrismaClient();

describe('AI Agent API Tests', () => {
  // Test user data
  const testUser = {
    id: '7d8f1e2a-9c0b-4f5a-b6c7-d8e9f0a1b2c3',
    email: 'agent_test_user@example.com',
    password: 'SecurePassword123!',
    name: 'Agent Test User'
  };

  // Token for authenticated requests
  let authToken: string;

  // Test vault data
  const testVault = {
    id: '',
    name: 'Agent Test Vault',
    description: 'Vault for agent testing',
    minDeposit: '100',
    maxCapacity: '1000000',
    performanceFee: 200, // 2%
    managementFee: 100, // 1%
  };

  // Test agent data
  const testAgent = {
    name: 'Test AI Agent',
    description: 'AI agent for testing',
    model: 'Ariadne-v1.0',
    strategyType: 'MOMENTUM',
    riskLevel: 3, // Medium risk
    maxPositionSize: '10000',
    maxDailyTrades: 10,
    status: 'INACTIVE'
  };

  let agentId: string;

  before(async () => {
    // Clean up any existing test data
    await prisma.aiAgent.deleteMany({
      where: {
        vault: {
          user: {
            email: testUser.email
          }
        }
      }
    });
    
    await prisma.vault.deleteMany({
      where: {
        user: {
          email: testUser.email
        }
      }
    });
    
    await prisma.user.deleteMany({
      where: {
        email: testUser.email
      }
    });

    // Create test user
    const hashedPassword = await hashPassword(testUser.password);
    await prisma.user.create({
      data: {
        id: testUser.id,
        email: testUser.email,
        password: hashedPassword,
        name: testUser.name,
        roles: {
          connect: { name: 'user' }
        }
      }
    });

    // Generate auth token
    authToken = jwt.sign(
      { id: testUser.id, email: testUser.email, roles: ['user'] },
      process.env.JWT_SECRET || 'minos-ai-secret-key',
      { expiresIn: '1h' }
    );

    // Create test vault
    const vault = await prisma.vault.create({
      data: {
        name: testVault.name,
        description: testVault.description,
        minDeposit: testVault.minDeposit,
        maxCapacity: testVault.maxCapacity,
        performanceFee: testVault.performanceFee,
        managementFee: testVault.managementFee,
        user: {
          connect: { id: testUser.id }
        }
      }
    });
    
    testVault.id = vault.id;
  });

  after(async () => {
    // Clean up test data
    await prisma.aiAgent.deleteMany({
      where: {
        vault: {
          user: {
            email: testUser.email
          }
        }
      }
    });
    
    await prisma.vault.deleteMany({
      where: {
        user: {
          email: testUser.email
        }
      }
    });
    
    await prisma.user.deleteMany({
      where: {
        email: testUser.email
      }
    });

    await prisma.$disconnect();
  });

  describe('POST /api/agents', () => {
    it('should create a new AI agent', async () => {
      const agentData = {
        ...testAgent,
        vaultId: testVault.id
      };

      const res = await request(app)
        .post('/api/agents')
        .set('Authorization', `Bearer ${authToken}`)
        .send(agentData);

      expect(res.status).to.equal(201);
      expect(res.body).to.have.property('id');
      expect(res.body.name).to.equal(testAgent.name);
      expect(res.body.model).to.equal(testAgent.model);
      expect(res.body.strategyType).to.equal(testAgent.strategyType);
      expect(res.body.vaultId).to.equal(testVault.id);
      
      // Save agent ID for later tests
      agentId = res.body.id;
    });

    it('should validate required fields', async () => {
      const res = await request(app)
        .post('/api/agents')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          // Missing required fields
          name: 'Incomplete Agent',
          vaultId: testVault.id
        });

      expect(res.status).to.equal(400);
      expect(res.body).to.have.property('errors');
    });

    it('should validate risk level range', async () => {
      const invalidRisk = {
        ...testAgent,
        vaultId: testVault.id,
        riskLevel: 6 // Out of range (should be 1-5)
      };

      const res = await request(app)
        .post('/api/agents')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidRisk);

      expect(res.status).to.equal(400);
      expect(res.body.message).to.include('risk level');
    });

    it('should require the user to own the vault', async () => {
      // Create another user
      const anotherUser = {
        email: 'another_agent_user@example.com',
        password: 'AnotherSecurePass123!'
      };

      await prisma.user.create({
        data: {
          email: anotherUser.email,
          password: await hashPassword(anotherUser.password),
          name: 'Another Agent User',
          roles: {
            connect: { name: 'user' }
          }
        }
      });

      const anotherToken = jwt.sign(
        { email: anotherUser.email, roles: ['user'] },
        process.env.JWT_SECRET || 'minos-ai-secret-key',
        { expiresIn: '1h' }
      );

      const res = await request(app)
        .post('/api/agents')
        .set('Authorization', `Bearer ${anotherToken}`)
        .send({
          ...testAgent,
          vaultId: testVault.id // This vault belongs to the original test user
        });

      expect(res.status).to.equal(403);

      // Clean up another user
      await prisma.user.deleteMany({
        where: { email: anotherUser.email }
      });
    });
  });

  describe('GET /api/agents', () => {
    it('should return all agents for the user', async () => {
      const res = await request(app)
        .get('/api/agents')
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      
      const createdAgent = res.body.find((a: any) => a.id === agentId);
      expect(createdAgent).to.exist;
      expect(createdAgent.name).to.equal(testAgent.name);
    });

    it('should filter agents by vault ID', async () => {
      const res = await request(app)
        .get(`/api/agents?vaultId=${testVault.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      expect(res.body[0].vaultId).to.equal(testVault.id);
    });

    it('should filter agents by status', async () => {
      const res = await request(app)
        .get('/api/agents?status=INACTIVE')
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      expect(res.body[0].status).to.equal('INACTIVE');
    });
  });

  describe('GET /api/agents/:id', () => {
    it('should return a specific agent', async () => {
      const res = await request(app)
        .get(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body.id).to.equal(agentId);
      expect(res.body.name).to.equal(testAgent.name);
      expect(res.body.model).to.equal(testAgent.model);
    });

    it('should return 404 for non-existent agent', async () => {
      const nonExistentId = '11111111-1111-1111-1111-111111111111';
      const res = await request(app)
        .get(`/api/agents/${nonExistentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(404);
    });
  });

  describe('PUT /api/agents/:id', () => {
    it('should update agent details', async () => {
      const updates = {
        name: 'Updated Agent Name',
        description: 'Updated agent description',
        riskLevel: 2,
        maxDailyTrades: 15
      };

      const res = await request(app)
        .put(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(updates);

      expect(res.status).to.equal(200);
      expect(res.body.name).to.equal(updates.name);
      expect(res.body.description).to.equal(updates.description);
      expect(res.body.riskLevel).to.equal(updates.riskLevel);
      expect(res.body.maxDailyTrades).to.equal(updates.maxDailyTrades);
      // Fields not in updates should remain unchanged
      expect(res.body.model).to.equal(testAgent.model);
    });

    it('should validate update data', async () => {
      const invalidUpdates = {
        maxDailyTrades: -5 // Invalid negative value
      };

      const res = await request(app)
        .put(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidUpdates);

      expect(res.status).to.equal(400);
    });
  });

  describe('POST /api/agents/:id/activate', () => {
    it('should activate an agent', async () => {
      const res = await request(app)
        .post(`/api/agents/${agentId}/activate`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body.status).to.equal('ACTIVE');
    });

    it('should require the agent to be in INACTIVE state', async () => {
      // Agent is already active, should fail
      const res = await request(app)
        .post(`/api/agents/${agentId}/activate`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(400);
      expect(res.body.message).to.include('already active');
    });
  });

  describe('POST /api/agents/:id/deactivate', () => {
    it('should deactivate an agent', async () => {
      const res = await request(app)
        .post(`/api/agents/${agentId}/deactivate`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body.status).to.equal('INACTIVE');
    });
  });

  describe('GET /api/agents/:id/performance', () => {
    it('should return performance metrics for an agent', async () => {
      // Create some test performance data
      await prisma.agentPerformance.create({
        data: {
          agentId: agentId,
          totalPnl: '1500',
          winRate: 0.65,
          totalTrades: 20,
          successfulTrades: 13,
          failedTrades: 7,
          sharpeRatio: 1.8,
          maxDrawdown: -0.12,
          averageTradeSize: '750',
          timestamp: new Date()
        }
      });

      const res = await request(app)
        .get(`/api/agents/${agentId}/performance`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.have.property('totalPnl');
      expect(res.body).to.have.property('winRate');
      expect(res.body.totalTrades).to.equal(20);
      expect(res.body.successfulTrades).to.equal(13);
    });
  });

  describe('GET /api/agents/:id/trades', () => {
    it('should return trades executed by the agent', async () => {
      // Create a test trade
      await prisma.agentTrade.create({
        data: {
          agentId: agentId,
          vaultId: testVault.id,
          asset: 'BTC-USD',
          direction: 'BUY',
          amount: '0.5',
          price: '45000',
          status: 'EXECUTED',
          pnl: '500',
          executedAt: new Date()
        }
      });

      const res = await request(app)
        .get(`/api/agents/${agentId}/trades`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      expect(res.body[0]).to.have.property('asset');
      expect(res.body[0].asset).to.equal('BTC-USD');
      expect(res.body[0].direction).to.equal('BUY');
    });

    it('should filter trades by date range', async () => {
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      // Filter to include today's trade
      const res = await request(app)
        .get(`/api/agents/${agentId}/trades?startDate=${yesterday.toISOString()}&endDate=${tomorrow.toISOString()}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);

      // Filter to exclude today's trade
      const pastRes = await request(app)
        .get(`/api/agents/${agentId}/trades?endDate=${yesterday.toISOString()}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(pastRes.status).to.equal(200);
      expect(pastRes.body).to.be.an('array');
      expect(pastRes.body.length).to.equal(0);
    });
  });

  describe('POST /api/agents/:id/parameters', () => {
    it('should update agent strategy parameters', async () => {
      const strategyParams = {
        indicators: ['sma', 'ema', 'rsi', 'macd'],
        timeframes: ['1m', '5m', '15m', '1h'],
        entryThreshold: 0.7,
        exitThreshold: 0.3
      };

      const res = await request(app)
        .post(`/api/agents/${agentId}/parameters`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({ parameters: strategyParams });

      expect(res.status).to.equal(200);
      expect(res.body).to.have.property('parameters');
      expect(res.body.parameters).to.deep.equal(strategyParams);
    });

    it('should validate parameter format', async () => {
      const invalidParams = {
        indicators: 'invalid-format', // Should be an array
        timeframes: ['invalid-timeframe'],
        entryThreshold: 2.0 // Out of range (should be 0-1)
      };

      const res = await request(app)
        .post(`/api/agents/${agentId}/parameters`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({ parameters: invalidParams });

      expect(res.status).to.equal(400);
    });
  });

  describe('POST /api/agents/:id/backtest', () => {
    it('should run a backtest for the agent strategy', async () => {
      const backtestParams = {
        startDate: '2023-01-01T00:00:00Z',
        endDate: '2023-01-31T23:59:59Z',
        assets: ['BTC-USD', 'ETH-USD'],
        initialCapital: '10000'
      };

      const res = await request(app)
        .post(`/api/agents/${agentId}/backtest`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(backtestParams);

      expect(res.status).to.equal(202); // Accepted (async operation)
      expect(res.body).to.have.property('backtestId');
    });

    it('should validate backtest parameters', async () => {
      const invalidParams = {
        startDate: 'invalid-date',
        endDate: '2023-01-31T23:59:59Z',
        assets: [],
        initialCapital: '-1000'
      };

      const res = await request(app)
        .post(`/api/agents/${agentId}/backtest`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidParams);

      expect(res.status).to.equal(400);
    });
  });

  describe('GET /api/agents/:id/backtest/:backtestId', () => {
    let backtestId: string;

    before(async () => {
      // Create a test backtest
      const backtest = await prisma.backtest.create({
        data: {
          agentId: agentId,
          status: 'COMPLETED',
          startDate: new Date('2023-01-01'),
          endDate: new Date('2023-01-31'),
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
          profitFactor: 1.67
        }
      });
      
      backtestId = backtest.id;
    });

    it('should return backtest results for the agent', async () => {
      const res = await request(app)
        .get(`/api/agents/${agentId}/backtest/${backtestId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body.id).to.equal(backtestId);
      expect(res.body.agentId).to.equal(agentId);
      expect(res.body.status).to.equal('COMPLETED');
      expect(res.body.totalReturn).to.equal(0.25);
      expect(res.body.totalTrades).to.equal(45);
    });

    it('should return 404 for non-existent backtest', async () => {
      const nonExistentId = '22222222-2222-2222-2222-222222222222';
      const res = await request(app)
        .get(`/api/agents/${agentId}/backtest/${nonExistentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(404);
    });
  });

  describe('GET /api/agents/models', () => {
    it('should return available AI models', async () => {
      const res = await request(app)
        .get('/api/agents/models')
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      expect(res.body[0]).to.have.property('id');
      expect(res.body[0]).to.have.property('name');
      expect(res.body[0]).to.have.property('description');
    });
  });

  describe('GET /api/agents/strategies', () => {
    it('should return available strategy types', async () => {
      const res = await request(app)
        .get('/api/agents/strategies')
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      expect(res.body).to.be.an('array');
      expect(res.body.length).to.be.at.least(1);
      expect(res.body[0]).to.have.property('id');
      expect(res.body[0]).to.have.property('name');
      expect(res.body[0]).to.have.property('description');
    });
  });

  describe('DELETE /api/agents/:id', () => {
    it('should not allow deletion of active agents', async () => {
      // First, activate the agent
      await request(app)
        .post(`/api/agents/${agentId}/activate`)
        .set('Authorization', `Bearer ${authToken}`);

      // Try to delete
      const res = await request(app)
        .delete(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(400);
      expect(res.body.message).to.include('active');
    });

    it('should delete an inactive agent', async () => {
      // First, deactivate the agent
      await request(app)
        .post(`/api/agents/${agentId}/deactivate`)
        .set('Authorization', `Bearer ${authToken}`);
      
      // Now delete
      const res = await request(app)
        .delete(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(res.status).to.equal(200);
      
      // Verify agent is deleted
      const checkRes = await request(app)
        .get(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${authToken}`);
        
      expect(checkRes.status).to.equal(404);
    });
  });
});