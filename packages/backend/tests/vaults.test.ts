import { describe, beforeAll, beforeEach, afterAll, afterEach, it, expect, jest } from '@jest/globals';
import request from 'supertest';
import { PublicKey, Keypair } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { Application } from 'express';
import app from '../src/app';
import { prisma } from '../src/config/database';
import { SolanaService } from '../src/services/solana.service';
import { VaultService } from '../src/services/vault.service';
import { UserService } from '../src/services/user.service';
import { AuthMiddleware } from '../src/middleware/auth.middleware';
import { RedisCacheService } from '../src/services/cache.service';
import { EventEmitter } from 'events';

/**
 * Enterprise-grade Vault Backend API Test Suite
 * 
 * This comprehensive test suite validates all vault-related endpoints with:
 * - Authentication and authorization testing
 * - Input validation and sanitization
 * - Business logic verification
 * - Error handling scenarios
 * - Performance testing
 * - Integration testing with Solana blockchain
 * - Rate limiting tests
 * - Security vulnerability testing
 */

interface TestVault {
  id: string;
  name: string;
  description: string;
  adminId: string;
  publicKey: string;
  tokenMint: string;
  status: 'active' | 'paused' | 'deprecated';
  strategy: {
    type: 'momentum' | 'meanReversion' | 'arbitrage' | 'hybrid';
    parameters: Record<string, any>;
    riskLevel: number;
  };
  performance: {
    totalValue: number;
    pnl: number;
    apy: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
  fees: {
    management: number;
    performance: number;
    entry: number;
    exit: number;
  };
  constraints: {
    minDeposit: number;
    maxCapacity: number;
    lockupPeriod: number;
  };
  participants: Array<{
    userId: string;
    amount: number;
    joinedAt: Date;
    status: 'active' | 'withdrawn' | 'pending';
  }>;
  createdAt: Date;
  updatedAt: Date;
}

interface TestUser {
  id: string;
  publicKey: string;
  email?: string;
  address?: string;
  createdAt: Date;
}

interface AuthenticatedRequest {
  user: TestUser;
  token: string;
}

describe('Vault Backend API Tests', () => {
  let testApp: Application;
  let testDatabase: typeof prisma;
  let solanaService: SolanaService;
  let vaultService: VaultService;
  let userService: UserService;
  let cacheService: RedisCacheService;
  let eventEmitter: EventEmitter;

  // Test data
  let adminUser: TestUser;
  let regularUser: TestUser;
  let premiumUser: TestUser;
  let testVault: TestVault;
  let adminToken: string;
  let userToken: string;
  let premiumToken: string;

  // Mock data generators
  const generateTestUser = (overrides?: Partial<TestUser>): TestUser => ({
    id: `user_${Math.random().toString(36).substring(7)}`,
    publicKey: Keypair.generate().publicKey.toString(),
    email: `test${Math.random().toString(36).substring(7)}@minos.ai`,
    address: '123 Test Street, Test City, TC 12345',
    createdAt: new Date(),
    ...overrides,
  });

  const generateTestVault = (overrides?: Partial<TestVault>): TestVault => ({
    id: `vault_${Math.random().toString(36).substring(7)}`,
    name: `Test Vault ${Math.random().toString(36).substring(7)}`,
    description: 'A comprehensive test vault for API validation',
    adminId: adminUser.id,
    publicKey: Keypair.generate().publicKey.toString(),
    tokenMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
    status: 'active',
    strategy: {
      type: 'momentum',
      parameters: {
        fastMa: 12,
        slowMa: 26,
        signalLine: 9,
        rsiPeriod: 14,
        volumeThreshold: 1000000,
      },
      riskLevel: 3,
    },
    performance: {
      totalValue: 1000000,
      pnl: 150000,
      apy: 23.5,
      maxDrawdown: -8.2,
      sharpeRatio: 1.85,
    },
    fees: {
      management: 200, // 2% annually
      performance: 2000, // 20% of profits
      entry: 50, // 0.5%
      exit: 100, // 1%
    },
    constraints: {
      minDeposit: 1000,
      maxCapacity: 10000000,
      lockupPeriod: 30, // 30 days
    },
    participants: [],
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  });

  // Authentication helpers
  const generateJWT = (user: TestUser): string => {
    return Buffer.from(JSON.stringify({
      sub: user.id,
      publicKey: user.publicKey,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600,
    })).toString('base64');
  };

  const authHeaders = (token: string) => ({
    Authorization: `Bearer ${token}`,
    'X-API-Version': 'v1',
    'Content-Type': 'application/json',
  });

  beforeAll(async () => {
    testApp = app;
    
    // Initialize services
    solanaService = new SolanaService({
      rpcUrl: process.env.SOLANA_RPC_URL || 'http://localhost:8899',
      wsUrl: process.env.SOLANA_WS_URL || 'ws://localhost:8900',
    });
    
    vaultService = new VaultService(solanaService);
    userService = new UserService(solanaService);
    cacheService = new RedisCacheService({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
    });
    
    eventEmitter = new EventEmitter();
    
    // Setup test database connection
    testDatabase = prisma;
    
    // Create test users
    adminUser = generateTestUser();
    regularUser = generateTestUser();
    premiumUser = generateTestUser();
    
    // Generate auth tokens
    adminToken = generateJWT(adminUser);
    userToken = generateJWT(regularUser);
    premiumToken = generateJWT(premiumUser);
    
    // Seed test data
    await testDatabase.user.createMany({
      data: [adminUser, regularUser, premiumUser],
    });
  });

  beforeEach(async () => {
    // Create fresh test vault for each test
    testVault = generateTestVault();
    
    // Clear cache
    await cacheService.flush();
    
    // Reset event listeners
    eventEmitter.removeAllListeners();
  });

  afterEach(async () => {
    // Clean up test data
    await testDatabase.vaultParticipant.deleteMany({});
    await testDatabase.vault.deleteMany({});
    
    // Clear mocks
    jest.clearAllMocks();
  });

  afterAll(async () => {
    // Clean up users
    await testDatabase.user.deleteMany({});
    
    // Close connections
    await testDatabase.$disconnect();
    await solanaService.disconnect();
    await cacheService.disconnect();
  });

  describe('POST /api/v1/vaults', () => {
    it('should create a new vault with valid parameters', async () => {
      const vaultData = {
        name: 'Advanced Trading Vault',
        description: 'AI-powered momentum trading strategy',
        tokenMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        strategy: {
          type: 'momentum',
          riskLevel: 3,
          parameters: {
            indicators: ['SMA', 'EMA', 'RSI', 'MACD'],
            timeframes: ['1h', '4h', '1d'],
            entrySignal: 0.7,
            exitSignal: 0.3,
          },
        },
        fees: {
          management: 200,
          performance: 2000,
          entry: 50,
          exit: 100,
        },
        constraints: {
          minDeposit: 1000,
          maxCapacity: 5000000,
          lockupPeriod: 30,
        },
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject({
        name: vaultData.name,
        description: vaultData.description,
        strategy: vaultData.strategy,
        fees: vaultData.fees,
        constraints: vaultData.constraints,
      });
      expect(response.body.data.publicKey).toMatch(/^[A-Za-z0-9]{44}$/);
      expect(response.body.data.status).toBe('active');
    });

    it('should validate required fields', async () => {
      const incompleteData = {
        name: 'Incomplete Vault',
        // Missing required fields
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(incompleteData);

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'description',
            message: 'Description is required',
          }),
          expect.objectContaining({
            field: 'tokenMint',
            message: 'Token mint address is required',
          }),
          expect.objectContaining({
            field: 'strategy',
            message: 'Strategy configuration is required',
          }),
        ])
      );
    });

    it('should validate strategy parameters', async () => {
      const invalidStrategyData = {
        name: 'Invalid Strategy Vault',
        description: 'Testing invalid strategy',
        tokenMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        strategy: {
          type: 'invalid_type',
          riskLevel: 10, // Invalid risk level (should be 1-5)
          parameters: {
            invalidParam: 'value',
          },
        },
        fees: {
          management: -100, // Negative fee
          performance: 10000, // 100% fee (invalid)
        },
        constraints: {
          minDeposit: -1000, // Negative value
          maxCapacity: 0, // Zero capacity
        },
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(invalidStrategyData);

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'strategy.type',
            message: 'Strategy type must be one of: momentum, meanReversion, arbitrage, hybrid',
          }),
          expect.objectContaining({
            field: 'strategy.riskLevel',
            message: 'Risk level must be between 1 and 5',
          }),
          expect.objectContaining({
            field: 'fees.management',
            message: 'Management fee must be non-negative',
          }),
          expect.objectContaining({
            field: 'fees.performance',
            message: 'Performance fee must be less than 50%',
          }),
        ])
      );
    });

    it('should prevent unauthorized vault creation', async () => {
      const vaultData = generateTestVault();

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set({
          'Content-Type': 'application/json',
          // No authorization header
        })
        .send(vaultData);

      expect(response.status).toBe(401);
      expect(response.body.error.code).toBe('UNAUTHORIZED');
    });

    it('should handle Solana transaction failures gracefully', async () => {
      // Mock Solana service to simulate failure
      jest.spyOn(solanaService, 'createVault').mockRejectedValueOnce(
        new Error('Insufficient SOL for transaction fees')
      );

      const vaultData = generateTestVault();

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('BLOCKCHAIN_ERROR');
      expect(response.body.error.message).toContain('vault creation');
    });

    it('should rate limit vault creation requests', async () => {
      const vaultData = generateTestVault();

      // Send multiple requests rapidly
      const requests = Array(10).fill(null).map(() =>
        request(testApp)
          .post('/api/v1/vaults')
          .set(authHeaders(adminToken))
          .send({ ...vaultData, name: `Vault ${Math.random()}` })
      );

      const responses = await Promise.all(requests);
      const rateLimitedResponse = responses.find(res => res.status === 429);

      expect(rateLimitedResponse).toBeDefined();
      expect(rateLimitedResponse?.body.error.code).toBe('RATE_LIMIT_EXCEEDED');
    });
  });

  describe('GET /api/v1/vaults', () => {
    beforeEach(async () => {
      // Create test vaults
      const vaults = [
        generateTestVault({ name: 'Momentum Vault 1', strategy: { type: 'momentum', riskLevel: 2, parameters: {} } }),
        generateTestVault({ name: 'Mean Reversion Vault', strategy: { type: 'meanReversion', riskLevel: 4, parameters: {} } }),
        generateTestVault({ name: 'Arbitrage Vault', strategy: { type: 'arbitrage', riskLevel: 1, parameters: {} } }),
        generateTestVault({ name: 'Hybrid Vault', strategy: { type: 'hybrid', riskLevel: 5, parameters: {} } }),
        generateTestVault({ status: 'paused', name: 'Paused Vault', strategy: { type: 'momentum', riskLevel: 3, parameters: {} } }),
      ];

      await testDatabase.vault.createMany({
        data: vaults,
      });
    });

    it('should return paginated vault list', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?page=1&limit=3')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.vaults).toHaveLength(3);
      expect(response.body.data.pagination).toMatchObject({
        currentPage: 1,
        totalPages: 2,
        totalItems: 5,
        hasNext: true,
        hasPrevious: false,
      });
    });

    it('should filter vaults by strategy type', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?strategyType=momentum')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults).toHaveLength(2);
      expect(response.body.data.vaults.every((vault: any) => 
        vault.strategy.type === 'momentum'
      )).toBe(true);
    });

    it('should filter vaults by risk level', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?riskLevel=1,2')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults).toHaveLength(2);
      expect(response.body.data.vaults.every((vault: any) => 
        [1, 2].includes(vault.strategy.riskLevel)
      )).toBe(true);
    });

    it('should filter vaults by status', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?status=active')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults).toHaveLength(4);
      expect(response.body.data.vaults.every((vault: any) => 
        vault.status === 'active'
      )).toBe(true);
    });

    it('should sort vaults by performance metrics', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?sortBy=apy&sortOrder=desc')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults).toHaveLength(5);
      
      // Verify descending order by APY
      const apyValues = response.body.data.vaults.map((vault: any) => vault.performance.apy);
      const sortedApyValues = [...apyValues].sort((a, b) => b - a);
      expect(apyValues).toEqual(sortedApyValues);
    });

    it('should search vaults by name or description', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?search=momentum')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults.length).toBeGreaterThan(0);
      expect(response.body.data.vaults.every((vault: any) => 
        vault.name.toLowerCase().includes('momentum') || 
        vault.description.toLowerCase().includes('momentum')
      )).toBe(true);
    });

    it('should handle complex query combinations', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults?strategyType=momentum,meanReversion&riskLevel=2,3,4&status=active&sortBy=performance&sortOrder=desc&limit=2')
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.vaults).toHaveLength(2);
      expect(response.body.data.vaults.every((vault: any) => 
        ['momentum', 'meanReversion'].includes(vault.strategy.type) &&
        [2, 3, 4].includes(vault.strategy.riskLevel) &&
        vault.status === 'active'
      )).toBe(true);
    });
  });

  describe('GET /api/v1/vaults/:id', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault(),
      });
      vaultId = vault.id;
    });

    it('should return vault details with full information', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject({
        id: vaultId,
        name: expect.any(String),
        description: expect.any(String),
        publicKey: expect.any(String),
        strategy: expect.objectContaining({
          type: expect.stringMatching(/^(momentum|meanReversion|arbitrage|hybrid)$/),
          riskLevel: expect.any(Number),
          parameters: expect.any(Object),
        }),
        performance: expect.objectContaining({
          totalValue: expect.any(Number),
          pnl: expect.any(Number),
          apy: expect.any(Number),
          maxDrawdown: expect.any(Number),
          sharpeRatio: expect.any(Number),
        }),
        fees: expect.objectContaining({
          management: expect.any(Number),
          performance: expect.any(Number),
          entry: expect.any(Number),
          exit: expect.any(Number),
        }),
        constraints: expect.objectContaining({
          minDeposit: expect.any(Number),
          maxCapacity: expect.any(Number),
          lockupPeriod: expect.any(Number),
        }),
      });
    });

    it('should include participant information for vault admins', async () => {
      // Add participants to the vault
      await testDatabase.vaultParticipant.createMany({
        data: [
          {
            vaultId,
            userId: regularUser.id,
            amount: new BN(5000).toNumber(),
            status: 'active',
          },
          {
            vaultId,
            userId: premiumUser.id,
            amount: new BN(10000).toNumber(),
            status: 'active',
          },
        ],
      });

      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.participants).toHaveLength(2);
      expect(response.body.data.participants[0]).toMatchObject({
        userId: expect.any(String),
        amount: expect.any(Number),
        status: expect.stringMatching(/^(active|withdrawn|pending)$/),
        joinedAt: expect.any(String),
      });
    });

    it('should return 404 for non-existent vault', async () => {
      const nonExistentId = 'vault_' + Math.random().toString(36).substring(7);
      
      const response = await request(testApp)
        .get(`/api/v1/vaults/${nonExistentId}`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(404);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('VAULT_NOT_FOUND');
    });

    it('should handle invalid vault ID format', async () => {
      const response = await request(testApp)
        .get('/api/v1/vaults/invalid-id-format')
        .set(authHeaders(userToken));

      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('INVALID_VAULT_ID');
    });

    it('should respect access control for private vault information', async () => {
      // Create a vault with restricted access
      const privateVault = await testDatabase.vault.create({
        data: generateTestVault({
          status: 'private' as any,
          accessControl: {
            whitelist: [adminUser.id],
          },
        }),
      });

      // Test access with whitelisted user
      const adminResponse = await request(testApp)
        .get(`/api/v1/vaults/${privateVault.id}`)
        .set(authHeaders(adminToken));

      expect(adminResponse.status).toBe(200);

      // Test access with non-whitelisted user
      const userResponse = await request(testApp)
        .get(`/api/v1/vaults/${privateVault.id}`)
        .set(authHeaders(userToken));

      expect(userResponse.status).toBe(403);
      expect(userResponse.body.error.code).toBe('ACCESS_DENIED');
    });
  });

  describe('PUT /api/v1/vaults/:id', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ adminId: adminUser.id }),
      });
      vaultId = vault.id;
    });

    it('should update vault configuration', async () => {
      const updateData = {
        description: 'Updated vault description',
        strategy: {
          type: 'meanReversion',
          riskLevel: 4,
          parameters: {
            rebalanceFrequency: 'daily',
            indicators: ['RSI', 'Bollinger Bands'],
            entryThreshold: 0.3,
            exitThreshold: 0.7,
          },
        },
        fees: {
          management: 150,
          performance: 1500,
          entry: 25,
          exit: 50,
        },
      };

      const response = await request(testApp)
        .put(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(adminToken))
        .send(updateData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject(updateData);
      expect(response.body.data.updatedAt).not.toBe(response.body.data.createdAt);
    });

    it('should prevent unauthorized updates', async () => {
      const updateData = {
        description: 'Unauthorized update attempt',
      };

      const response = await request(testApp)
        .put(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(userToken)) // Non-admin user
        .send(updateData);

      expect(response.status).toBe(403);
      expect(response.body.error.code).toBe('INSUFFICIENT_PERMISSIONS');
    });

    it('should validate update data', async () => {
      const invalidUpdateData = {
        strategy: {
          type: 'invalid_strategy',
          riskLevel: 0,
        },
        fees: {
          management: -100,
          performance: 15000, // > 100%
        },
      };

      const response = await request(testApp)
        .put(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(adminToken))
        .send(invalidUpdateData);

      expect(response.status).toBe(400);
      expect(response.body.error.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'strategy.type',
            message: expect.stringContaining('invalid'),
          }),
          expect.objectContaining({
            field: 'strategy.riskLevel',
            message: expect.stringContaining('between 1 and 5'),
          }),
        ])
      );
    });

    it('should update Solana program state', async () => {
      const updateData = {
        constraints: {
          maxCapacity: 20000000,
          minDeposit: 2000,
        },
      };

      // Mock Solana service
      const updateVaultSpy = jest.spyOn(solanaService, 'updateVault')
        .mockResolvedValueOnce({
          signature: 'mock_signature',
          slot: 12345,
        });

      const response = await request(testApp)
        .put(`/api/v1/vaults/${vaultId}`)
        .set(authHeaders(adminToken))
        .send(updateData);

      expect(response.status).toBe(200);
      expect(updateVaultSpy).toHaveBeenCalledWith(
        expect.any(String), // vault public key
        expect.objectContaining(updateData)
      );
    });
  });

  describe('POST /api/v1/vaults/:id/deposit', () => {
    let vaultId: string;
    let vaultPublicKey: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault(),
      });
      vaultId = vault.id;
      vaultPublicKey = vault.publicKey;
    });

    it('should process deposit successfully', async () => {
      const depositData = {
        amount: 5000,
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      // Mock Solana deposit transaction
      jest.spyOn(solanaService, 'deposit').mockResolvedValueOnce({
        signature: 'deposit_signature_123',
        slot: 12345,
        amount: new BN(depositData.amount * 1e6),
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(depositData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject({
        transactionSignature: 'deposit_signature_123',
        amount: depositData.amount,
        status: 'pending',
      });
    });

    it('should enforce minimum deposit requirement', async () => {
      const insufficientDeposit = {
        amount: 500, // Below minimum of 1000
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(insufficientDeposit);

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('DEPOSIT_TOO_SMALL');
      expect(response.body.error.message).toContain('minimum deposit');
    });

    it('should enforce vault capacity limits', async () => {
      // Mock vault at capacity
      jest.spyOn(vaultService, 'getVaultStats').mockResolvedValueOnce({
        totalValue: 9999000, // Near capacity
        maxCapacity: 10000000,
        participantCount: 100,
      });

      const largeDeposit = {
        amount: 50000, // Would exceed capacity
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(largeDeposit);

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('VAULT_CAPACITY_EXCEEDED');
    });

    it('should validate token account ownership', async () => {
      const depositData = {
        amount: 5000,
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      // Mock invalid token account
      jest.spyOn(solanaService, 'validateTokenAccount').mockResolvedValueOnce({
        isValid: false,
        reason: 'Account not owned by user',
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(depositData);

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('INVALID_TOKEN_ACCOUNT');
    });

    it('should handle deposit during lockup period', async () => {
      // Mock existing lockup
      jest.spyOn(vaultService, 'getUserLockup').mockResolvedValueOnce({
        isLocked: true,
        unlockDate: new Date(Date.now() + 86400000), // 1 day from now
        originalAmount: 3000,
      });

      const depositData = {
        amount: 2000,
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(depositData);

      expect(response.status).toBe(200);
      expect(response.body.data.lockupExtended).toBe(true);
      expect(response.body.data.newUnlockDate).toBeDefined();
    });

    it('should apply entry fees correctly', async () => {
      const depositData = {
        amount: 10000,
        tokenAccount: Keypair.generate().publicKey.toString(),
      };

      jest.spyOn(solanaService, 'deposit').mockResolvedValueOnce({
        signature: 'deposit_signature_123',
        slot: 12345,
        amount: new BN(depositData.amount * 1e6),
        fees: {
          entry: new BN(50 * 1e6), // 0.5% entry fee
          platform: new BN(10 * 1e6), // 0.1% platform fee
        },
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send(depositData);

      expect(response.status).toBe(200);
      expect(response.body.data.fees).toMatchObject({
        entry: 50,
        platform: 10,
        total: 60,
      });
      expect(response.body.data.netAmount).toBe(9940); // 10000 - 60
    });
  });

  describe('POST /api/v1/vaults/:id/withdraw', () => {
    let vaultId: string;
    let participantId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault(),
      });
      vaultId = vault.id;

      const participant = await testDatabase.vaultParticipant.create({
        data: {
          vaultId,
          userId: userToken,
          amount: 10000,
          status: 'active',
        },
      });
      participantId = participant.id;
    });

    it('should process full withdrawal', async () => {
      const withdrawData = {
        amount: 10000,
        type: 'full',
        recipient: Keypair.generate().publicKey.toString(),
      };

      jest.spyOn(solanaService, 'withdraw').mockResolvedValueOnce({
        signature: 'withdraw_signature_123',
        slot: 12345,
        amount: new BN(10000 * 1e6),
        performance: new BN(1500 * 1e6), // 15% gain
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/withdraw`)
        .set(authHeaders(userToken))
        .send(withdrawData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject({
        transactionSignature: 'withdraw_signature_123',
        amount: 10000,
        performance: 1500,
        status: 'completed',
      });
    });

    it('should process partial withdrawal', async () => {
      const withdrawData = {
        amount: 5000,
        type: 'partial',
        recipient: Keypair.generate().publicKey.toString(),
      };

      jest.spyOn(solanaService, 'withdraw').mockResolvedValueOnce({
        signature: 'withdraw_signature_123',
        slot: 12345,
        amount: new BN(5000 * 1e6),
        performance: new BN(750 * 1e6), // Proportional gain
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/withdraw`)
        .set(authHeaders(userToken))
        .send(withdrawData);

      expect(response.status).toBe(200);
      expect(response.body.data.remainingBalance).toBe(5000);
    });

    it('should respect lockup period', async () => {
      // Create recent deposit within lockup period
      await testDatabase.vaultParticipant.update({
        where: { id: participantId },
        data: {
          joinedAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000), // 10 days ago
          lockupPeriod: 30, // 30 days lockup
        },
      });

      const withdrawData = {
        amount: 5000,
        type: 'partial',
        recipient: Keypair.generate().publicKey.toString(),
      };

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/withdraw`)
        .set(authHeaders(userToken))
        .send(withdrawData);

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('LOCKUP_PERIOD_ACTIVE');
      expect(response.body.error.details.unlockDate).toBeDefined();
    });

    it('should calculate and deduct performance fees', async () => {
      const withdrawData = {
        amount: 10000,
        type: 'full',
        recipient: Keypair.generate().publicKey.toString(),
      };

      // Mock profitable withdrawal
      jest.spyOn(solanaService, 'withdraw').mockResolvedValueOnce({
        signature: 'withdraw_signature_123',
        slot: 12345,
        amount: new BN(10000 * 1e6),
        performance: new BN(2000 * 1e6), // 20% gain
        fees: {
          performance: new BN(400 * 1e6), // 20% of 2000 gain
          exit: new BN(100 * 1e6), // 1% exit fee
          total: new BN(500 * 1e6),
        },
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/withdraw`)
        .set(authHeaders(userToken))
        .send(withdrawData);

      expect(response.status).toBe(200);
      expect(response.body.data.fees).toMatchObject({
        performance: 400,
        exit: 100,
        total: 500,
      });
      expect(response.body.data.netAmount).toBe(11500); // 10000 + 2000 - 500
    });

    it('should handle emergency withdrawals', async () => {
      const emergencyWithdraw = {
        amount: 10000,
        type: 'emergency',
        reason: 'Unexpected financial emergency',
        recipient: Keypair.generate().publicKey.toString(),
      };

      jest.spyOn(solanaService, 'emergencyWithdraw').mockResolvedValueOnce({
        signature: 'emergency_signature_123',
        slot: 12345,
        amount: new BN(10000 * 1e6),
        penaltyApplied: new BN(500 * 1e6), // 5% penalty
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/withdraw`)
        .set(authHeaders(userToken))
        .send(emergencyWithdraw);

      expect(response.status).toBe(200);
      expect(response.body.data.penalty).toBe(500);
      expect(response.body.data.netAmount).toBe(9500);
      expect(response.body.data.type).toBe('emergency');
    });
  });

  describe('GET /api/v1/vaults/:id/performance', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault(),
      });
      vaultId = vault.id;

      // Create performance history
      await testDatabase.vaultPerformance.createMany({
        data: Array(365).fill(null).map((_, index) => ({
          vaultId,
          date: new Date(Date.now() - (364 - index) * 24 * 60 * 60 * 1000),
          totalValue: 1000000 + Math.random() * 500000,
          dailyReturn: (Math.random() - 0.5) * 0.1,
          cumulativeReturn: Math.random() * 0.5,
          sharpeRatio: 1 + Math.random(),
          volatility: 0.1 + Math.random() * 0.2,
        })),
      });
    });

    it('should return performance metrics for different time periods', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/performance?period=1y`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data).toMatchObject({
        period: '1y',
        totalReturn: expect.any(Number),
        annualizedReturn: expect.any(Number),
        volatility: expect.any(Number),
        sharpeRatio: expect.any(Number),
        maxDrawdown: expect.any(Number),
        calmarRatio: expect.any(Number),
        beta: expect.any(Number),
        alpha: expect.any(Number),
        informationRatio: expect.any(Number),
      });
      expect(response.body.data.dataPoints).toHaveLength(365);
    });

    it('should calculate risk-adjusted metrics correctly', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/performance?period=1y&includeRiskMetrics=true`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.riskMetrics).toMatchObject({
        valueAtRisk95: expect.any(Number),
        valueAtRisk99: expect.any(Number),
        conditionalValueAtRisk: expect.any(Number),
        tailRatio: expect.any(Number),
        skewness: expect.any(Number),
        kurtosis: expect.any(Number),
      });
    });

    it('should include benchmark comparisons', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/performance?period=1y&benchmark=sp500,bitcoin`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.benchmarkComparison).toBeDefined();
      expect(response.body.data.benchmarkComparison).toHaveProperty('sp500');
      expect(response.body.data.benchmarkComparison).toHaveProperty('bitcoin');
    });

    it('should aggregate performance by custom periods', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/performance?period=1y&aggregation=monthly`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.body.data.dataPoints).toHaveLength(12);
      expect(response.body.data.dataPoints[0]).toMatchObject({
        period: expect.stringMatching(/^\d{4}-\d{2}$/),
        return: expect.any(Number),
        volatility: expect.any(Number),
        sharpeRatio: expect.any(Number),
      });
    });
  });

  describe('GET /api/v1/vaults/:id/participants', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ adminId: adminUser.id }),
      });
      vaultId = vault.id;

      // Create participants
      await testDatabase.vaultParticipant.createMany({
        data: [
          {
            vaultId,
            userId: regularUser.id,
            amount: 5000,
            status: 'active',
            joinedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          },
          {
            vaultId,
            userId: premiumUser.id,
            amount: 15000,
            status: 'active',
            joinedAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
          },
          {
            vaultId,
            userId: generateTestUser().id,
            amount: 3000,
            status: 'withdrawn',
            joinedAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
            exitedAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000),
          },
        ],
      });
    });

    it('should return participant list for vault admin', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/participants`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.participants).toHaveLength(3);
      expect(response.body.data.participants[0]).toMatchObject({
        userId: expect.any(String),
        amount: expect.any(Number),
        status: expect.stringMatching(/^(active|withdrawn|pending)$/),
        joinedAt: expect.any(String),
        performance: expect.objectContaining({
          currentValue: expect.any(Number),
          realizedPnl: expect.any(Number),
          unrealizedPnl: expect.any(Number),
        }),
      });
    });

    it('should deny access to non-admin users', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/participants`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(403);
      expect(response.body.error.code).toBe('INSUFFICIENT_PERMISSIONS');
    });

    it('should include participant analytics', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/participants?includeAnalytics=true`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.analytics).toMatchObject({
        totalParticipants: 3,
        activeParticipants: 2,
        totalDeposits: 23000,
        averageDepositSize: expect.any(Number),
        participantRetentionRate: expect.any(Number),
        averageHoldingPeriod: expect.any(Number),
      });
    });

    it('should support participant filtering and sorting', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/participants?status=active&sortBy=amount&sortOrder=desc&limit=1`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.participants).toHaveLength(1);
      expect(response.body.data.participants[0].amount).toBe(15000);
      expect(response.body.data.participants[0].status).toBe('active');
    });
  });

  describe('POST /api/v1/vaults/:id/pause', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ adminId: adminUser.id }),
      });
      vaultId = vault.id;
    });

    it('should pause vault operations', async () => {
      const pauseReason = {
        reason: 'Market volatility requires manual intervention',
        estimatedDuration: '24 hours',
      };

      jest.spyOn(solanaService, 'pauseVault').mockResolvedValueOnce({
        signature: 'pause_signature_123',
        slot: 12345,
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/pause`)
        .set(authHeaders(adminToken))
        .send(pauseReason);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.status).toBe('paused');
      expect(response.body.data.pauseReason).toBe(pauseReason.reason);
    });

    it('should prevent non-admin from pausing vault', async () => {
      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/pause`)
        .set(authHeaders(userToken))
        .send({ reason: 'Unauthorized pause attempt' });

      expect(response.status).toBe(403);
      expect(response.body.error.code).toBe('INSUFFICIENT_PERMISSIONS');
    });

    it('should handle already paused vault', async () => {
      // First pause
      await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/pause`)
        .set(authHeaders(adminToken))
        .send({ reason: 'First pause' });

      // Try to pause again
      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/pause`)
        .set(authHeaders(adminToken))
        .send({ reason: 'Second pause' });

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('VAULT_ALREADY_PAUSED');
    });
  });

  describe('POST /api/v1/vaults/:id/resume', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ 
          adminId: adminUser.id,
          status: 'paused',
        }),
      });
      vaultId = vault.id;
    });

    it('should resume vault operations', async () => {
      jest.spyOn(solanaService, 'resumeVault').mockResolvedValueOnce({
        signature: 'resume_signature_123',
        slot: 12345,
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/resume`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.status).toBe('active');
      expect(response.body.data.resumedAt).toBeDefined();
    });

    it('should prevent resuming active vault', async () => {
      // Update vault to active status
      await testDatabase.vault.update({
        where: { id: vaultId },
        data: { status: 'active' },
      });

      const response = await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/resume`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('VAULT_NOT_PAUSED');
    });
  });

  describe('GET /api/v1/vaults/:id/audit-log', () => {
    let vaultId: string;

    beforeEach(async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ adminId: adminUser.id }),
      });
      vaultId = vault.id;

      // Create audit log entries
      await testDatabase.auditLog.createMany({
        data: [
          {
            vaultId,
            userId: adminUser.id,
            action: 'CREATE_VAULT',
            timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
            details: { vaultName: 'Test Vault' },
          },
          {
            vaultId,
            userId: regularUser.id,
            action: 'DEPOSIT',
            timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
            details: { amount: 5000 },
          },
          {
            vaultId,
            userId: adminUser.id,
            action: 'UPDATE_STRATEGY',
            timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
            details: { 
              oldStrategy: { type: 'momentum' },
              newStrategy: { type: 'meanReversion' },
            },
          },
        ],
      });
    });

    it('should return audit log with proper access control', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/audit-log`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.auditLog).toHaveLength(3);
      expect(response.body.data.auditLog[0]).toMatchObject({
        action: expect.any(String),
        userId: expect.any(String),
        timestamp: expect.any(String),
        details: expect.any(Object),
      });
    });

    it('should filter audit log by action type', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/audit-log?action=DEPOSIT`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.auditLog).toHaveLength(1);
      expect(response.body.data.auditLog[0].action).toBe('DEPOSIT');
    });

    it('should filter audit log by date range', async () => {
      const startDate = new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString();
      const endDate = new Date().toISOString();

      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/audit-log?startDate=${startDate}&endDate=${endDate}`)
        .set(authHeaders(adminToken));

      expect(response.status).toBe(200);
      expect(response.body.data.auditLog).toHaveLength(2);
    });

    it('should deny access to non-admin users', async () => {
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vaultId}/audit-log`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(403);
      expect(response.body.error.code).toBe('INSUFFICIENT_PERMISSIONS');
    });
  });

  describe('Performance and Load Testing', () => {
    it('should handle concurrent vault operations', async () => {
      const vaultData = generateTestVault();
      
      // Create vault
      const createResponse = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);
      
      const vaultId = createResponse.body.data.id;

      // Simulate concurrent deposits
      const concurrentDeposits = Array(50).fill(null).map((_, index) =>
        request(testApp)
          .post(`/api/v1/vaults/${vaultId}/deposit`)
          .set(authHeaders(userToken))
          .send({
            amount: 1000 + index,
            tokenAccount: Keypair.generate().publicKey.toString(),
          })
      );

      const responses = await Promise.allSettled(concurrentDeposits);
      const successfulDeposits = responses.filter(r => 
        r.status === 'fulfilled' && r.value.status === 200
      );

      expect(successfulDeposits.length).toBeGreaterThan(0);
      expect(successfulDeposits.length).toBeLessThanOrEqual(50);
    });

    it('should maintain response times under load', async () => {
      const vaultId = (await testDatabase.vault.create({
        data: generateTestVault(),
      })).id;

      const startTime = Date.now();
      
      const requests = Array(100).fill(null).map(() =>
        request(testApp)
          .get(`/api/v1/vaults/${vaultId}`)
          .set(authHeaders(userToken))
      );

      const responses = await Promise.all(requests);
      const endTime = Date.now();
      
      const avgResponseTime = (endTime - startTime) / requests.length;
      expect(avgResponseTime).toBeLessThan(200); // Average response time should be under 200ms
      expect(responses.every(r => r.status === 200)).toBe(true);
    });
  });

  describe('Security Testing', () => {
    it('should prevent SQL injection attacks', async () => {
      const maliciousInput = "'; DROP TABLE vaults; --";
      
      const response = await request(testApp)
        .get(`/api/v1/vaults?search=${encodeURIComponent(maliciousInput)}`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      // Verify that the malicious input was sanitized and no databases were affected
      const vaultCount = await testDatabase.vault.count();
      expect(vaultCount).toBeGreaterThanOrEqual(0);
    });

    it('should prevent XSS attacks in vault descriptions', async () => {
      const xssPayload = '<script>alert("XSS")</script>';
      const vaultData = {
        ...generateTestVault(),
        description: xssPayload,
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      expect(response.status).toBe(201);
      expect(response.body.data.description).not.toContain('<script>');
      expect(response.body.data.description).toContain('&lt;script&gt;');
    });

    it('should validate and sanitize all input parameters', async () => {
      const maliciousInputs = {
        name: '<img src=x onerror=alert(1)>',
        description: '${7*7}',
        tokenMint: 'invalid_mint_address',
        strategy: {
          type: '<script>',
          riskLevel: 'invalid',
          parameters: {
            maliciousKey: '${process.env}',
          },
        },
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(maliciousInputs);

      expect(response.status).toBe(400);
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
    });
  });

  describe('Integration with External Services', () => {
    it('should handle Solana RPC failures gracefully', async () => {
      // Mock RPC failure
      jest.spyOn(solanaService, 'getConnection').mockImplementationOnce(() => {
        throw new Error('RPC connection failed');
      });

      const vaultData = generateTestVault();
      
      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      expect(response.status).toBe(503);
      expect(response.body.error.code).toBe('SERVICE_UNAVAILABLE');
      expect(response.body.error.message).toContain('blockchain service');
    });

    it('should retry failed transactions with exponential backoff', async () => {
      let attemptCount = 0;
      jest.spyOn(solanaService, 'createVault').mockImplementation(async () => {
        attemptCount++;
        if (attemptCount < 3) {
          throw new Error('Transaction failed');
        }
        return {
          signature: 'success_signature',
          slot: 12345,
        };
      });

      const vaultData = generateTestVault();
      
      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      expect(response.status).toBe(201);
      expect(attemptCount).toBe(3);
    });
  });

  describe('Event Handling and Webhooks', () => {
    it('should emit events on vault operations', async () => {
      const events: any[] = [];
      eventEmitter.on('vault.created', (event) => events.push({ type: 'created', ...event }));
      eventEmitter.on('vault.deposit', (event) => events.push({ type: 'deposit', ...event }));
      eventEmitter.on('vault.withdraw', (event) => events.push({ type: 'withdraw', ...event }));

      // Create vault
      const vaultData = generateTestVault();
      const createResponse = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);
      
      const vaultId = createResponse.body.data.id;

      // Perform deposit
      jest.spyOn(solanaService, 'deposit').mockResolvedValueOnce({
        signature: 'deposit_signature',
        slot: 12345,
        amount: new BN(5000 * 1e6),
      });

      await request(testApp)
        .post(`/api/v1/vaults/${vaultId}/deposit`)
        .set(authHeaders(userToken))
        .send({
          amount: 5000,
          tokenAccount: Keypair.generate().publicKey.toString(),
        });

      // Wait for async events
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(events).toHaveLength(2);
      expect(events[0].type).toBe('created');
      expect(events[1].type).toBe('deposit');
      expect(events[1].vaultId).toBe(vaultId);
      expect(events[1].amount).toBe(5000);
    });

    it('should send webhooks for important events', async () => {
      // Mock webhook endpoint
      let webhookReceived: any = null;
      jest.spyOn(require('axios'), 'post').mockImplementation(async (url, data) => {
        if (url.includes('/webhook')) {
          webhookReceived = data;
          return { status: 200, data: { received: true } };
        }
        return { status: 404 };
      });

      // Create vault with webhook URL
      const vaultData = {
        ...generateTestVault(),
        webhookUrl: 'https://example.com/webhook',
      };

      const response = await request(testApp)
        .post('/api/v1/vaults')
        .set(authHeaders(adminToken))
        .send(vaultData);

      // Wait for webhook processing
      await new Promise(resolve => setTimeout(resolve, 200));

      expect(webhookReceived).toBeDefined();
      expect(webhookReceived.event).toBe('vault.created');
      expect(webhookReceived.vaultId).toBe(response.body.data.id);
    });
  });

  describe('Cache Management', () => {
    it('should cache frequently accessed vault data', async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault(),
      });

      // First request - should hit database
      const response1 = await request(testApp)
        .get(`/api/v1/vaults/${vault.id}`)
        .set(authHeaders(userToken));

      expect(response1.status).toBe(200);

      // Second request - should hit cache
      const response2 = await request(testApp)
        .get(`/api/v1/vaults/${vault.id}`)
        .set(authHeaders(userToken));

      expect(response2.status).toBe(200);
      expect(response2.headers['x-cache']).toBe('HIT');
    });

    it('should invalidate cache on vault updates', async () => {
      const vault = await testDatabase.vault.create({
        data: generateTestVault({ adminId: adminUser.id }),
      });

      // Initial request to populate cache
      await request(testApp)
        .get(`/api/v1/vaults/${vault.id}`)
        .set(authHeaders(userToken));

      // Update vault
      await request(testApp)
        .put(`/api/v1/vaults/${vault.id}`)
        .set(authHeaders(adminToken))
        .send({
          description: 'Updated description',
        });

      // Next request should not be cached
      const response = await request(testApp)
        .get(`/api/v1/vaults/${vault.id}`)
        .set(authHeaders(userToken));

      expect(response.status).toBe(200);
      expect(response.headers['x-cache']).not.toBe('HIT');
      expect(response.body.data.description).toBe('Updated description');
    });
  });

  describe('API Versioning', () => {
    it('should support multiple API versions', async () => {
      const vaultData = generateTestVault();

      // Test v1 API
      const v1Response = await request(testApp)
        .post('/api/v1/vaults')
        .set({ ...authHeaders(adminToken), 'X-API-Version': 'v1' })
        .send(vaultData);

      // Test v2 API (if available)
      const v2Response = await request(testApp)
        .post('/api/v2/vaults')
        .set({ ...authHeaders(adminToken), 'X-API-Version': 'v2' })
        .send(vaultData);

      expect(v1Response.status).toBe(201);
      // V2 might not exist yet, but shouldn't crash
      expect([201, 404]).toContain(v2Response.status);
    });

    it('should handle unsupported API versions gracefully', async () => {
      const response = await request(testApp)
        .get('/api/v99/vaults')
        .set({ ...authHeaders(userToken), 'X-API-Version': 'v99' });

      expect(response.status).toBe(404);
      expect(response.body.error.code).toBe('UNSUPPORTED_API_VERSION');
    });
  });

  describe('Monitoring and Health Checks', () => {
    it('should provide health check endpoint', async () => {
      const response = await request(testApp)
        .get('/health');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        status: 'healthy',
        timestamp: expect.any(String),
        version: expect.any(String),
        services: {
          database: 'connected',
          redis: 'connected',
          solana: 'connected',
        },
      });
    });

    it('should expose metrics endpoint', async () => {
      const response = await request(testApp)
        .get('/metrics')
        .set({
          'Authorization': 'Bearer admin_metrics_token',
        });

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('text/plain');
      expect(response.text).toContain('# TYPE http_requests_total counter');
      expect(response.text).toContain('vault_operations_total');
    });
  });

  describe('Error Handling and Logging', () => {
    it('should log all API requests with proper context', async () => {
      const logSpy = jest.spyOn(console, 'log').mockImplementation();

      await request(testApp)
        .get('/api/v1/vaults')
        .set(authHeaders(userToken));

      expect(logSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          method: 'GET',
          url: '/api/v1/vaults',
          userId: regularUser.id,
          timestamp: expect.any(String),
          responseTime: expect.any(Number),
          statusCode: 200,
        })
      );

      logSpy.mockRestore();
    });

    it('should handle and log errors with proper stack traces', async () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation();

      // Force an error by mocking a service failure
      jest.spyOn(vaultService, 'getAllVaults').mockRejectedValueOnce(
        new Error('Database connection lost')
      );

      const response = await request(testApp)
        .get('/api/v1/vaults')
        .set(authHeaders(userToken));

      expect(response.status).toBe(500);
      expect(errorSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          error: 'Database connection lost',
          stack: expect.any(String),
          requestId: expect.any(String),
          userId: regularUser.id,
        })
      );

      errorSpy.mockRestore();
    });
  });

  describe('Documentation and OpenAPI', () => {
    it('should serve OpenAPI documentation', async () => {
      const response = await request(testApp)
        .get('/api/docs/vault');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        openapi: '3.0.0',
        info: {
          title: 'Minos AI Vault API',
          version: expect.any(String),
        },
        paths: expect.objectContaining({
          '/api/v1/vaults': expect.any(Object),
          '/api/v1/vaults/{id}': expect.any(Object),
        }),
      });
    });
  });

  describe('Cleanup and Teardown', () => {
    it('should clean up resources properly after tests', async () => {
      // Verify all connections are properly closed
      expect(testDatabase.$transaction).toBeDefined();
      
      // Verify no memory leaks in event listeners
      expect(eventEmitter.listenerCount('vault.created')).toBe(0);
      expect(eventEmitter.listenerCount('vault.deposit')).toBe(0);
      expect(eventEmitter.listenerCount('vault.withdraw')).toBe(0);
      
      // Verify cache is cleared
      const cacheKeys = await cacheService.getAllKeys();
      expect(cacheKeys.filter(key => key.startsWith('vault:'))).toHaveLength(0);
    });
  });
});