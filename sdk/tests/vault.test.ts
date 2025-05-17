import { expect } from 'chai';
import nock from 'nock';
import { MinosClient } from '../src/client';
import { VaultClient } from '../src/vault';
import { Vault, VaultTransaction, VaultPerformance } from '../src/types';

describe('VaultClient', () => {
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
  const testVault: Vault = {
    id: 'v-123456',
    name: 'Test Vault',
    description: 'A test vault',
    minDeposit: '100',
    maxCapacity: '1000000',
    performanceFee: 200, // 2%
    managementFee: 100, // 1%
    userId: testUser.id,
    balance: '5000',
    totalDeposited: '5000',
    totalWithdrawn: '0',
    status: 'ACTIVE',
    createdAt: '2025-01-01T00:00:00Z',
    updatedAt: '2025-01-01T00:00:00Z'
  };
  
  // Test transaction data
  const testTransaction: VaultTransaction = {
    id: 't-123456',
    vaultId: testVault.id,
    type: 'DEPOSIT',
    amount: '1000',
    token: 'USDC',
    status: 'COMPLETED',
    transactionHash: '0x1234567890abcdef',
    createdAt: '2025-01-02T00:00:00Z',
    completedAt: '2025-01-02T00:01:00Z'
  };
  
  // Test performance data
  const testPerformance: VaultPerformance = {
    vaultId: testVault.id,
    totalValueLocked: '5000',
    allTimeReturn: 0.15,
    dailyReturns: [0.01, 0.02, -0.005, 0.015],
    sharpeRatio: 1.8,
    maxDrawdown: -0.05,
    volatility: 0.02,
    assetAllocation: [
      { asset: 'BTC-USD', percentage: 0.4 },
      { asset: 'ETH-USD', percentage: 0.3 },
      { asset: 'SOL-USD', percentage: 0.2 },
      { asset: 'USDC', percentage: 0.1 }
    ],
    updatedAt: '2025-01-03T00:00:00Z'
  };

  let minosClient: MinosClient;
  let vaultClient: VaultClient;

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
    
    // Create vault client
    vaultClient = new VaultClient(minosClient);
    
    // Reset any previous nock interceptors for API endpoints
    nock.cleanAll();
  });

  afterEach(() => {
    // Ensure all nock interceptors have been used
    expect(nock.isDone()).to.be.true;
  });

  describe('List Vaults', () => {
    it('should list user vaults', async () => {
      nock(API_URL)
        .get('/vaults')
        .reply(200, [testVault]);

      const vaults = await vaultClient.listVaults();
      
      expect(vaults).to.be.an('array');
      expect(vaults.length).to.equal(1);
      expect(vaults[0].id).to.equal(testVault.id);
    });

    it('should support pagination', async () => {
      nock(API_URL)
        .get('/vaults?page=2&limit=10')
        .reply(200, [testVault]);

      const vaults = await vaultClient.listVaults({ page: 2, limit: 10 });
      
      expect(vaults).to.be.an('array');
      expect(vaults.length).to.equal(1);
    });

    it('should support filtering', async () => {
      nock(API_URL)
        .get('/vaults?status=ACTIVE')
        .reply(200, [testVault]);

      const vaults = await vaultClient.listVaults({ status: 'ACTIVE' });
      
      expect(vaults).to.be.an('array');
      expect(vaults.length).to.equal(1);
      expect(vaults[0].status).to.equal('ACTIVE');
    });
  });

  describe('Get Vault', () => {
    it('should get a vault by ID', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}`)
        .reply(200, testVault);

      const vault = await vaultClient.getVault(testVault.id);
      
      expect(vault).to.deep.equal(testVault);
    });

    it('should throw an error if vault not found', async () => {
      const nonExistentId = 'v-nonexistent';
      
      nock(API_URL)
        .get(`/vaults/${nonExistentId}`)
        .reply(404, {
          message: 'Vault not found'
        });

      try {
        await vaultClient.getVault(nonExistentId);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Vault not found');
        expect(error.status).to.equal(404);
      }
    });
  });

  describe('Create Vault', () => {
    it('should create a new vault', async () => {
      const vaultData = {
        name: 'New Vault',
        description: 'A new test vault',
        minDeposit: '100',
        maxCapacity: '1000000',
        performanceFee: 200,
        managementFee: 100
      };
      
      const newVault = {
        ...testVault,
        id: 'v-new123',
        name: vaultData.name,
        description: vaultData.description
      };
      
      nock(API_URL)
        .post('/vaults', vaultData)
        .reply(201, newVault);

      const vault = await vaultClient.createVault(vaultData);
      
      expect(vault).to.deep.equal(newVault);
    });

    it('should validate required fields', async () => {
      const invalidData = {
        // Missing name
        description: 'Invalid vault'
      };
      
      nock(API_URL)
        .post('/vaults', invalidData)
        .reply(400, {
          message: 'Validation failed',
          errors: ['name is required']
        });

      try {
        await vaultClient.createVault(invalidData);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Validation failed');
        expect(error.status).to.equal(400);
        expect(error.errors).to.deep.equal(['name is required']);
      }
    });
  });

  describe('Update Vault', () => {
    it('should update a vault', async () => {
      const updateData = {
        name: 'Updated Vault',
        description: 'Updated description'
      };
      
      const updatedVault = {
        ...testVault,
        name: updateData.name,
        description: updateData.description,
        updatedAt: '2025-01-04T00:00:00Z'
      };
      
      nock(API_URL)
        .put(`/vaults/${testVault.id}`, updateData)
        .reply(200, updatedVault);

      const vault = await vaultClient.updateVault(testVault.id, updateData);
      
      expect(vault).to.deep.equal(updatedVault);
    });

    it('should throw an error if vault not found', async () => {
      const nonExistentId = 'v-nonexistent';
      const updateData = { name: 'Updated Vault' };
      
      nock(API_URL)
        .put(`/vaults/${nonExistentId}`, updateData)
        .reply(404, {
          message: 'Vault not found'
        });

      try {
        await vaultClient.updateVault(nonExistentId, updateData);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Vault not found');
        expect(error.status).to.equal(404);
      }
    });
  });

  describe('Delete Vault', () => {
    it('should delete a vault', async () => {
      nock(API_URL)
        .delete(`/vaults/${testVault.id}`)
        .reply(200, {
          success: true,
          message: 'Vault deleted successfully'
        });

      await vaultClient.deleteVault(testVault.id);
      // If no error is thrown, the test passes
    });

    it('should throw an error if vault cannot be deleted', async () => {
      nock(API_URL)
        .delete(`/vaults/${testVault.id}`)
        .reply(400, {
          message: 'Cannot delete vault with active deposits'
        });

      try {
        await vaultClient.deleteVault(testVault.id);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Cannot delete vault with active deposits');
        expect(error.status).to.equal(400);
      }
    });
  });

  describe('Vault Transactions', () => {
    it('should list vault transactions', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}/transactions`)
        .reply(200, [testTransaction]);

      const transactions = await vaultClient.listTransactions(testVault.id);
      
      expect(transactions).to.be.an('array');
      expect(transactions.length).to.equal(1);
      expect(transactions[0].id).to.equal(testTransaction.id);
    });

    it('should filter transactions by type', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}/transactions?type=DEPOSIT`)
        .reply(200, [testTransaction]);

      const transactions = await vaultClient.listTransactions(testVault.id, { type: 'DEPOSIT' });
      
      expect(transactions).to.be.an('array');
      expect(transactions.length).to.equal(1);
      expect(transactions[0].type).to.equal('DEPOSIT');
    });

    it('should deposit funds to a vault', async () => {
      const depositData = {
        amount: '1000',
        token: 'USDC'
      };
      
      nock(API_URL)
        .post(`/vaults/${testVault.id}/deposit`, depositData)
        .reply(200, testTransaction);

      const transaction = await vaultClient.deposit(testVault.id, depositData.amount, depositData.token);
      
      expect(transaction).to.deep.equal(testTransaction);
    });

    it('should withdraw funds from a vault', async () => {
      const withdrawData = {
        amount: '500',
        token: 'USDC'
      };
      
      const withdrawTransaction = {
        ...testTransaction,
        id: 't-withdraw123',
        type: 'WITHDRAWAL',
        amount: withdrawData.amount
      };
      
      nock(API_URL)
        .post(`/vaults/${testVault.id}/withdraw`, withdrawData)
        .reply(200, withdrawTransaction);

      const transaction = await vaultClient.withdraw(testVault.id, withdrawData.amount, withdrawData.token);
      
      expect(transaction).to.deep.equal(withdrawTransaction);
    });

    it('should throw an error if withdrawal exceeds balance', async () => {
      const withdrawData = {
        amount: '10000', // More than balance
        token: 'USDC'
      };
      
      nock(API_URL)
        .post(`/vaults/${testVault.id}/withdraw`, withdrawData)
        .reply(400, {
          message: 'Withdrawal amount exceeds available balance'
        });

      try {
        await vaultClient.withdraw(testVault.id, withdrawData.amount, withdrawData.token);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).to.equal('Withdrawal amount exceeds available balance');
        expect(error.status).to.equal(400);
      }
    });
  });

  describe('Vault Performance', () => {
    it('should get vault performance', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}/performance`)
        .reply(200, testPerformance);

      const performance = await vaultClient.getPerformance(testVault.id);
      
      expect(performance).to.deep.equal(testPerformance);
    });

    it('should filter performance by timeframe', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}/performance?timeframe=week`)
        .reply(200, {
          ...testPerformance,
          dailyReturns: testPerformance.dailyReturns.slice(0, 7) // Last week's returns
        });

      const performance = await vaultClient.getPerformance(testVault.id, { timeframe: 'week' });
      
      expect(performance).to.have.property('dailyReturns');
      expect(performance.dailyReturns.length).to.be.at.most(7);
    });
  });

  describe('Vault Agents', () => {
    it('should list agents associated with a vault', async () => {
      const testAgent = {
        id: 'a-123456',
        name: 'Test Agent',
        vaultId: testVault.id,
        model: 'Ariadne-v1.0',
        strategyType: 'MOMENTUM',
        status: 'ACTIVE'
      };
      
      nock(API_URL)
        .get(`/vaults/${testVault.id}/agents`)
        .reply(200, [testAgent]);

      const agents = await vaultClient.listAgents(testVault.id);
      
      expect(agents).to.be.an('array');
      expect(agents.length).to.equal(1);
      expect(agents[0].id).to.equal(testAgent.id);
      expect(agents[0].vaultId).to.equal(testVault.id);
    });
  });

  describe('Vault Stats', () => {
    it('should get vault statistics', async () => {
      const stats = {
        totalDeposits: 3,
        totalWithdrawals: 1,
        averageDepositAmount: '1666.67',
        averageWithdrawalAmount: '500.00',
        lastDeposit: {
          amount: '1000',
          date: '2025-01-02T00:00:00Z'
        },
        lastWithdrawal: {
          amount: '500',
          date: '2025-01-03T00:00:00Z'
        },
        activeAgents: 2,
        topPerformingAssets: [
          { asset: 'BTC-USD', return: 0.25 },
          { asset: 'ETH-USD', return: 0.15 }
        ]
      };
      
      nock(API_URL)
        .get(`/vaults/${testVault.id}/stats`)
        .reply(200, stats);

      const vaultStats = await vaultClient.getStats(testVault.id);
      
      expect(vaultStats).to.deep.equal(stats);
    });
  });

  describe('Vault Analytics', () => {
    it('should get vault analytics data', async () => {
      const analytics = {
        historicalBalance: [
          { date: '2025-01-01', balance: '3000' },
          { date: '2025-01-02', balance: '4000' },
          { date: '2025-01-03', balance: '5000' }
        ],
        returnAnalysis: {
          monthly: 0.05,
          quarterly: 0.12,
          yearly: 0.25,
          riskAdjusted: 0.18
        },
        riskMetrics: {
          volatility: 0.02,
          maxDrawdown: -0.05,
          downsideDeviation: 0.015,
          sortinoRatio: 2.1,
          calmarRatio: 3.2
        },
        correlations: [
          { asset: 'BTC-USD', correlation: 0.8 },
          { asset: 'ETH-USD', correlation: 0.7 },
          { asset: 'S&P 500', correlation: 0.3 }
        ]
      };
      
      nock(API_URL)
        .get(`/vaults/${testVault.id}/analytics`)
        .reply(200, analytics);

      const vaultAnalytics = await vaultClient.getAnalytics(testVault.id);
      
      expect(vaultAnalytics).to.deep.equal(analytics);
    });

    it('should support custom date ranges for analytics', async () => {
      const startDate = '2025-01-01';
      const endDate = '2025-01-03';
      
      nock(API_URL)
        .get(`/vaults/${testVault.id}/analytics?startDate=${startDate}&endDate=${endDate}`)
        .reply(200, {
          // Filtered analytics data
        });

      await vaultClient.getAnalytics(testVault.id, { startDate, endDate });
      // If no error is thrown, the test passes
    });
  });

  describe('Vault Export', () => {
    it('should export vault data', async () => {
      nock(API_URL)
        .get(`/vaults/${testVault.id}/export?format=json`)
        .reply(200, {
          vault: testVault,
          transactions: [testTransaction],
          performance: testPerformance,
          exportedAt: '2025-01-05T00:00:00Z'
        });

      const exportData = await vaultClient.exportData(testVault.id, 'json');
      
      expect(exportData).to.have.property('vault');
      expect(exportData).to.have.property('transactions');
      expect(exportData).to.have.property('performance');
      expect(exportData).to.have.property('exportedAt');
    });

    it('should support CSV export format', async () => {
      const csvData = 'id,name,description,balance\nv-123456,Test Vault,A test vault,5000';
      
      nock(API_URL)
        .get(`/vaults/${testVault.id}/export?format=csv`)
        .reply(200, csvData, {
          'Content-Type': 'text/csv'
        });

      const exportData = await vaultClient.exportData(testVault.id, 'csv');
      
      expect(exportData).to.equal(csvData);
    });
  });
});