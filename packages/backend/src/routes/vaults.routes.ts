/**
 * @fileoverview Vaults API Routes for Minos-AI DeFi Platform
 * @description Defines all HTTP endpoints for vault management and interaction
 * @author Minos-AI Engineering Team <engineering@minos-ai.io>
 * @copyright 2024 Minos-AI Labs, Inc.
 * @license MIT
 */

import { Router } from 'express';
import { body, param, query, validationResult } from 'express-validator';
import { vaultService } from '../services/vault.service';
import { createLogger } from '../utils/logger';
import { RoleGuard } from '../middlewares/role-guard.middleware';
import { asyncHandler } from '../middlewares/async-handler.middleware';
import { RateLimitMiddleware } from '../middlewares/rate-limit.middleware';
import { CacheMiddleware } from '../middlewares/cache.middleware';
import { TransactionLoggingMiddleware } from '../middlewares/transaction-logging.middleware';
import { 
  StrategyType, 
  VaultStatus,
  VaultCreateParams,
  VaultDepositParams,
  VaultWithdrawParams,
  StrategyExecutionParams 
} from '../interfaces/vault.interfaces';
import { HttpError } from '../utils/errors';

// Initialize router and logger
const router = Router();
const logger = createLogger('vaults-routes');

// Initialize middleware instances
const roleGuard = new RoleGuard();
const rateLimit = new RateLimitMiddleware();
const cache = new CacheMiddleware();
const transactionLogger = new TransactionLoggingMiddleware();

/**
 * @swagger
 * tags:
 *   name: Vaults
 *   description: Vault management and operations API
 */

/**
 * @swagger
 * /api/v1/vaults:
 *   get:
 *     summary: Get all vaults
 *     description: Retrieve a list of all vaults with optional filtering
 *     tags: [Vaults]
 *     parameters:
 *       - in: query
 *         name: includeInactive
 *         schema:
 *           type: boolean
 *           default: false
 *         description: Whether to include inactive vaults in the results
 *       - in: query
 *         name: tokenMint
 *         schema:
 *           type: string
 *         description: Filter vaults by token mint address
 *       - in: query
 *         name: strategyType
 *         schema:
 *           type: string
 *           enum: [YIELD_FARMING, LIQUIDITY_PROVIDING, LEVERAGE_TRADING, ARBITRAGE]
 *         description: Filter vaults by strategy type
 *     responses:
 *       200:
 *         description: A list of vaults
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/VaultResponse'
 *       500:
         description: Server error
         content:
           application/json:
             schema:
               $ref: '#/components/schemas/ErrorResponse'
 */
router.patch(
  '/:pubkey/status',
  [
    roleGuard.checkRole(['admin', 'vault_manager']),
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    body('status').isIn(Object.values(VaultStatus)).withMessage('Invalid vault status'),
    body('reason').optional().isString().trim().isLength({ max: 200 }).withMessage('Reason must be less than 200 characters'),
  ],
  transactionLogger.middleware('vault:update-status'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const pubkey = req.params.pubkey;
    const status = req.body.status as VaultStatus;
    const reason = req.body.reason || '';
    
    logger.info(`Updating status for vault ${pubkey} to ${status}. Reason: ${reason}`);
    
    // This endpoint would call a vault service method to update status
    // For now, we'll throw an error as this is not implemented yet
    throw new HttpError(501, 'Not Implemented', 'This endpoint is not yet implemented');
    
    // Once implemented, it would look like this:
    /*
    const signature = await vaultService.updateVaultStatus(pubkey, status, reason);
    
    // Invalidate cache for vault and list endpoints
    cache.invalidate(`/api/v1/vaults/${pubkey}`);
    cache.invalidate(`/api/v1/vaults/${pubkey}/stats`);
    cache.invalidate('/api/v1/vaults');
    
    return res.status(200).json({
      success: true,
      data: {
        transactionSignature: signature,
      },
    });
    */
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/fees:
 *   patch:
 *     summary: Update vault fees
 *     description: Update the management and performance fees for a vault
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               managementFeePercentage:
 *                 type: number
 *                 example: 1.5
 *               performanceFeePercentage:
 *                 type: number
 *                 example: 15.0
 *     responses:
 *       200:
 *         description: Vault fees updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     transactionSignature:
 *                       type: string
 *                       example: "5UfgccYvGrh2UfqCVEkZ8s4Vx9SRm2NUZbGwXbG4LrJMatHQNrJfYfWSybZMNqLnQtLdTBfYUjGvQDQBww4DDTyj"
 *       400:
 *         description: Invalid parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.patch(
  '/:pubkey/fees',
  [
    roleGuard.checkRole(['admin']), // Only admin can update fees
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    body('managementFeePercentage').optional().isFloat({ min: 0, max: 10 }).withMessage('Management fee must be between 0 and 10%'),
    body('performanceFeePercentage').optional().isFloat({ min: 0, max: 50 }).withMessage('Performance fee must be between 0 and 50%'),
  ],
  transactionLogger.middleware('vault:update-fees'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const pubkey = req.params.pubkey;
    const managementFeePercentage = req.body.managementFeePercentage;
    const performanceFeePercentage = req.body.performanceFeePercentage;
    
    if (!managementFeePercentage && !performanceFeePercentage) {
      throw new HttpError(400, 'Invalid Request', 'At least one fee parameter must be provided');
    }
    
    logger.info(`Updating fees for vault ${pubkey}. Management: ${managementFeePercentage}%, Performance: ${performanceFeePercentage}%`);
    
    // This endpoint would call a vault service method to update fees
    // For now, we'll throw an error as this is not implemented yet
    throw new HttpError(501, 'Not Implemented', 'This endpoint is not yet implemented');
    
    // Once implemented, it would look like this:
    /*
    const signature = await vaultService.updateVaultFees(pubkey, managementFeePercentage, performanceFeePercentage);
    
    // Invalidate cache for vault and list endpoints
    cache.invalidate(`/api/v1/vaults/${pubkey}`);
    cache.invalidate(`/api/v1/vaults/${pubkey}/stats`);
    
    return res.status(200).json({
      success: true,
      data: {
        transactionSignature: signature,
      },
    });
    */
  })
);

/**
 * @swagger
 * /api/v1/vaults/schedule-all-strategies:
 *   post:
 *     summary: Schedule strategy execution for all active vaults
 *     description: Trigger the scheduling of strategy execution for all active vaults
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     responses:
 *       200:
 *         description: Strategies scheduled successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: "Strategy execution scheduled for all active vaults"
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post(
  '/schedule-all-strategies',
  [
    roleGuard.checkRole(['admin', 'vault_manager']),
    rateLimit.limitByIp(1, 60 * 60), // 1 request per hour
  ],
  asyncHandler(async (req, res) => {
    logger.info('Scheduling strategy execution for all active vaults');
    
    await vaultService.scheduleAllVaultStrategyExecutions();
    
    return res.status(200).json({
      success: true,
      message: 'Strategy execution scheduled for all active vaults',
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/positions/{userPubkey}:
 *   get:
 *     summary: Get user position in a vault
 *     description: Retrieve details about a user's position in a specific vault
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *       - in: path
 *         name: userPubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the user
 *     responses:
 *       200:
 *         description: User position details
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/UserPositionResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       403:
 *         description: Forbidden - Cannot access another user's position
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault or user position not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get(
  '/:pubkey/positions/:userPubkey',
  [
    roleGuard.checkRole(['admin', 'user']),
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    param('userPubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid user public key'),
  ],
  cache.middleware(30), // Cache for 30 seconds
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const vaultPubkey = req.params.pubkey;
    const userPubkey = req.params.userPubkey;
    
    // Ensure user making request matches the userPubkey in the request
    // or has admin role (handled by role guard middleware)
    if (req.user.sub !== userPubkey && !req.user.roles.includes('admin')) {
      throw new HttpError(403, 'Forbidden', 'You can only view your own positions unless you are an admin');
    }
    
    logger.debug(`Getting position for user ${userPubkey} in vault ${vaultPubkey}`);
    
    // This endpoint would call a user position service method to get position details
    // For now, we'll throw an error as this is not implemented yet
    throw new HttpError(501, 'Not Implemented', 'This endpoint is not yet implemented');
    
    // Once implemented, it would look like this:
    /*
    const position = await userPositionService.getUserPosition(vaultPubkey, userPubkey);
    
    return res.status(200).json({
      success: true,
      data: position,
    });
    */
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/transactions:
 *   get:
 *     summary: Get vault transactions
 *     description: Retrieve a list of transactions for a specific vault
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           default: 1
 *         description: Page number for pagination
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *         description: Number of items per page
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [DEPOSIT, WITHDRAW, STRATEGY_EXECUTION]
 *         description: Filter transactions by type
 *       - in: query
 *         name: startDate
 *         schema:
 *           type: string
 *           format: date
 *         description: Filter transactions by start date (ISO format)
 *       - in: query
 *         name: endDate
 *         schema:
 *           type: string
 *           format: date
 *         description: Filter transactions by end date (ISO format)
 *     responses:
 *       200:
 *         description: Transaction list
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/TransactionResponse'
 *                 pagination:
 *                   type: object
 *                   properties:
 *                     total:
 *                       type: integer
 *                       example: 85
 *                     page:
 *                       type: integer
 *                       example: 1
 *                     limit:
 *                       type: integer
 *                       example: 20
 *                     pages:
 *                       type: integer
 *                       example: 5
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get(
  '/:pubkey/transactions',
  [
    roleGuard.checkRole(['admin', 'vault_manager', 'user']),
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    query('page').optional().isInt({ min: 1 }).toInt(),
    query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
    query('type').optional().isIn(['DEPOSIT', 'WITHDRAW', 'STRATEGY_EXECUTION']),
    query('startDate').optional().isISO8601(),
    query('endDate').optional().isISO8601(),
  ],
  cache.middleware(60), // Cache for 1 minute
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const vaultPubkey = req.params.pubkey;
    const page = req.query.page ? parseInt(req.query.page as string) : 1;
    const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
    const type = req.query.type as string;
    const startDate = req.query.startDate ? new Date(req.query.startDate as string) : undefined;
    const endDate = req.query.endDate ? new Date(req.query.endDate as string) : undefined;
    
    logger.debug(`Getting transactions for vault ${vaultPubkey}`);
    
    // This endpoint would call a transaction service method to get transaction history
    // For now, we'll throw an error as this is not implemented yet
    throw new HttpError(501, 'Not Implemented', 'This endpoint is not yet implemented');
    
    // Once implemented, it would look like this:
    /*
    const { transactions, total, pages } = await transactionService.getVaultTransactions(
      vaultPubkey,
      page,
      limit,
      type,
      startDate,
      endDate
    );
    
    return res.status(200).json({
      success: true,
      data: transactions,
      pagination: {
        total,
        page,
        limit,
        pages,
      },
    });
    */
  })
);

export { router as vaultRoutes };
*         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get(
  '/',
  [
    query('includeInactive').optional().isBoolean().toBoolean(),
    query('tokenMint').optional().isString(),
    query('strategyType').optional().isIn(Object.values(StrategyType)),
  ],
  cache.middleware(300), // Cache for 5 minutes
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const includeInactive = req.query.includeInactive as boolean || false;
    const tokenMint = req.query.tokenMint as string;
    const strategyType = req.query.strategyType as StrategyType;

    logger.debug(`Getting all vaults. includeInactive=${includeInactive}, tokenMint=${tokenMint}, strategyType=${strategyType}`);
    
    let vaults = await vaultService.getAllVaults(includeInactive);
    
    // Apply filters if provided
    if (tokenMint) {
      vaults = vaults.filter(vault => vault.account.tokenMint === tokenMint);
    }
    
    if (strategyType) {
      vaults = vaults.filter(vault => vault.account.strategyType === strategyType);
    }
    
    return res.status(200).json({
      success: true,
      data: vaults,
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}:
 *   get:
 *     summary: Get vault by pubkey
 *     description: Retrieve detailed information about a specific vault
 *     tags: [Vaults]
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     responses:
 *       200:
 *         description: Vault details
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/VaultResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get(
  '/:pubkey',
  [
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
  ],
  cache.middleware(60), // Cache for 1 minute
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const pubkey = req.params.pubkey;
    logger.debug(`Getting vault with pubkey: ${pubkey}`);
    
    const vault = await vaultService.getVault(pubkey);
    
    return res.status(200).json({
      success: true,
      data: vault,
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults:
 *   post:
 *     summary: Create a new vault
 *     description: Create a new investment vault with specified parameters
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - tokenMint
 *               - strategyType
 *               - managementFeePercentage
 *               - performanceFeePercentage
 *             properties:
 *               name:
 *                 type: string
 *                 example: "BTC Yield Farming Vault"
 *               tokenMint:
 *                 type: string
 *                 example: "So11111111111111111111111111111111111111112"
 *               strategyType:
 *                 type: string
 *                 enum: [YIELD_FARMING, LIQUIDITY_PROVIDING, LEVERAGE_TRADING, ARBITRAGE]
 *                 example: "YIELD_FARMING"
 *               managementFeePercentage:
 *                 type: number
 *                 example: 2.0
 *               performanceFeePercentage:
 *                 type: number
 *                 example: 20.0
 *               description:
 *                 type: string
 *                 example: "A vault focused on yield farming strategies for BTC"
 *     responses:
 *       201:
 *         description: Vault created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/VaultResponse'
 *       400:
 *         description: Invalid parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post(
  '/',
  [
    roleGuard.checkRole(['admin', 'vault_manager']),
    rateLimit.limitByIp(5, 60 * 60), // 5 requests per hour
    body('name').isString().trim().isLength({ min: 3, max: 50 }).withMessage('Name must be between 3 and 50 characters'),
    body('tokenMint').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana token mint address'),
    body('strategyType').isIn(Object.values(StrategyType)).withMessage('Invalid strategy type'),
    body('managementFeePercentage').isFloat({ min: 0, max: 10 }).withMessage('Management fee must be between 0 and 10%'),
    body('performanceFeePercentage').isFloat({ min: 0, max: 50 }).withMessage('Performance fee must be between 0 and 50%'),
    body('description').optional().isString().trim().isLength({ max: 500 }).withMessage('Description must be less than 500 characters'),
  ],
  transactionLogger.middleware('vault:create'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    logger.info(`Creating new vault with name: ${req.body.name}`);
    
    const params: VaultCreateParams = {
      name: req.body.name,
      tokenMint: req.body.tokenMint,
      strategyType: req.body.strategyType as StrategyType,
      managementFeePercentage: req.body.managementFeePercentage,
      performanceFeePercentage: req.body.performanceFeePercentage,
      description: req.body.description,
    };
    
    const vault = await vaultService.createVault(params);
    
    // Clear cache for vault list endpoint
    cache.invalidate('/api/v1/vaults');
    
    return res.status(201).json({
      success: true,
      data: vault,
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/deposit:
 *   post:
 *     summary: Deposit tokens into a vault
 *     description: Deposit tokens into a specified vault
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - userPubkey
 *               - amount
 *             properties:
 *               userPubkey:
 *                 type: string
 *                 example: "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
 *               amount:
 *                 type: number
 *                 example: 100.5
 *     responses:
 *       200:
 *         description: Deposit processed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     transactionSignature:
 *                       type: string
 *                       example: "5UfgccYvGrh2UfqCVEkZ8s4Vx9SRm2NUZbGwXbG4LrJMatHQNrJfYfWSybZMNqLnQtLdTBfYUjGvQDQBww4DDTyj"
 *       400:
 *         description: Invalid parameters or vault is inactive
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post(
  '/:pubkey/deposit',
  [
    roleGuard.checkRole(['admin', 'user']),
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    body('userPubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid user public key'),
    body('amount').isFloat({ min: 0.000001 }).withMessage('Amount must be greater than 0.000001'),
  ],
  transactionLogger.middleware('vault:deposit'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const vaultPubkey = req.params.pubkey;
    const userPubkey = req.body.userPubkey;
    const amount = req.body.amount;
    
    logger.info(`Processing deposit of ${amount} to vault ${vaultPubkey} for user ${userPubkey}`);
    
    // Ensure user making request matches the userPubkey in the request
    // or has admin/manager role (handled by role guard middleware)
    if (req.user.sub !== userPubkey && !req.user.roles.includes('admin') && !req.user.roles.includes('vault_manager')) {
      throw new HttpError(403, 'Forbidden', 'You can only deposit to your own wallet unless you are an admin or vault manager');
    }
    
    const params: VaultDepositParams = {
      vaultPubkey,
      userPubkey,
      amount,
    };
    
    const signature = await vaultService.deposit(params);
    
    // Invalidate cache for vault stats and details
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}`);
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}/stats`);
    
    return res.status(200).json({
      success: true,
      data: {
        transactionSignature: signature,
      },
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/withdraw:
 *   post:
 *     summary: Withdraw tokens from a vault
 *     description: Withdraw tokens from a specified vault
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - userPubkey
 *               - amount
 *             properties:
 *               userPubkey:
 *                 type: string
 *                 example: "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
 *               amount:
 *                 type: number
 *                 example: 50.25
 *     responses:
 *       200:
 *         description: Withdrawal processed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     transactionSignature:
 *                       type: string
 *                       example: "5UfgccYvGrh2UfqCVEkZ8s4Vx9SRm2NUZbGwXbG4LrJMatHQNrJfYfWSybZMNqLnQtLdTBfYUjGvQDQBww4DDTyj"
 *       400:
 *         description: Invalid parameters or insufficient balance
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post(
  '/:pubkey/withdraw',
  [
    roleGuard.checkRole(['admin', 'user']),
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    body('userPubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid user public key'),
    body('amount').isFloat({ min: 0.000001 }).withMessage('Amount must be greater than 0.000001'),
  ],
  transactionLogger.middleware('vault:withdraw'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const vaultPubkey = req.params.pubkey;
    const userPubkey = req.body.userPubkey;
    const amount = req.body.amount;
    
    logger.info(`Processing withdrawal of ${amount} from vault ${vaultPubkey} for user ${userPubkey}`);
    
    // Ensure user making request matches the userPubkey in the request
    // or has admin/manager role (handled by role guard middleware)
    if (req.user.sub !== userPubkey && !req.user.roles.includes('admin') && !req.user.roles.includes('vault_manager')) {
      throw new HttpError(403, 'Forbidden', 'You can only withdraw from your own wallet unless you are an admin or vault manager');
    }
    
    const params: VaultWithdrawParams = {
      vaultPubkey,
      userPubkey,
      amount,
    };
    
    const signature = await vaultService.withdraw(params);
    
    // Invalidate cache for vault stats and details
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}`);
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}/stats`);
    
    return res.status(200).json({
      success: true,
      data: {
        transactionSignature: signature,
      },
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/execute-strategy:
 *   post:
 *     summary: Execute vault investment strategy
 *     description: Trigger the execution of a vault's investment strategy
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               forceExecution:
 *                 type: boolean
 *                 default: false
 *                 description: Force strategy execution even if risk thresholds are exceeded
 *               additionalParams:
 *                 type: object
 *                 description: Additional strategy-specific parameters
 *                 properties:
 *                   targetApy:
 *                     type: number
 *                     example: 8.5
 *                   leverage:
 *                     type: number
 *                     example: 2.0
 *     responses:
 *       200:
 *         description: Strategy executed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     transactionSignature:
 *                       type: string
 *                       example: "5UfgccYvGrh2UfqCVEkZ8s4Vx9SRm2NUZbGwXbG4LrJMatHQNrJfYfWSybZMNqLnQtLdTBfYUjGvQDQBww4DDTyj"
 *       400:
 *         description: Invalid parameters or vault is inactive
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       409:
 *         description: Strategy execution already in progress
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post(
  '/:pubkey/execute-strategy',
  [
    roleGuard.checkRole(['admin', 'vault_manager']),
    rateLimit.limitByResource('vault-strategy', 1, 5 * 60), // 1 execution per 5 minutes per vault
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
    body('forceExecution').optional().isBoolean(),
    body('additionalParams').optional().isObject(),
    body('additionalParams.targetApy').optional().isFloat({ min: 0 }),
    body('additionalParams.leverage').optional().isFloat({ min: 1, max: 10 }),
  ],
  transactionLogger.middleware('vault:execute-strategy'),
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const vaultPubkey = req.params.pubkey;
    const forceExecution = req.body.forceExecution || false;
    const additionalParams = req.body.additionalParams;
    
    logger.info(`Executing strategy for vault ${vaultPubkey}. forceExecution=${forceExecution}`);
    
    const params: StrategyExecutionParams = {
      vaultPubkey,
      forceExecution,
      additionalParams,
    };
    
    const signature = await vaultService.executeStrategy(params);
    
    // Invalidate cache for vault stats and details
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}`);
    cache.invalidate(`/api/v1/vaults/${vaultPubkey}/stats`);
    
    return res.status(200).json({
      success: true,
      data: {
        transactionSignature: signature,
      },
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/stats:
 *   get:
 *     summary: Get vault statistics
 *     description: Retrieve detailed statistics and performance metrics for a vault
 *     tags: [Vaults]
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     responses:
 *       200:
 *         description: Vault statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/VaultStatsResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get(
  '/:pubkey/stats',
  [
    param('pubkey').isString().matches(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/).withMessage('Invalid Solana public key'),
  ],
  cache.middleware(60), // Cache for 1 minute
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      throw new HttpError(400, 'Validation Error', errors.array());
    }

    const pubkey = req.params.pubkey;
    logger.debug(`Getting statistics for vault: ${pubkey}`);
    
    const stats = await vaultService.getVaultStats(pubkey);
    
    return res.status(200).json({
      success: true,
      data: stats,
    });
  })
);

/**
 * @swagger
 * /api/v1/vaults/{pubkey}/status:
 *   patch:
 *     summary: Update vault status
 *     description: Update the status of a vault (activate, pause, or decommission)
 *     tags: [Vaults]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: pubkey
 *         required: true
 *         schema:
 *           type: string
 *         description: Public key of the vault
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - status
 *             properties:
 *               status:
 *                 type: string
 *                 enum: [ACTIVE, PAUSED, DECOMMISSIONED]
 *                 example: "PAUSED"
 *               reason:
 *                 type: string
 *                 example: "Pausing for maintenance and strategy adjustment"
 *     responses:
 *       200:
 *         description: Vault status updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     transactionSignature:
 *                       type: string
 *                       example: "5UfgccYvGrh2UfqCVEkZ8s4Vx9SRm2NUZbGwXbG4LrJMatHQNrJfYfWSybZMNqLnQtLdTBfYUjGvQDQBww4DDTyj"
 *       400:
 *         description: Invalid parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       404:
 *         description: Vault not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500: