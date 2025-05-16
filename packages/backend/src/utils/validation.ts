/**
 * Validation Utilities for Minos-AI DeFi Platform
 * Comprehensive validation system for API requests, data integrity,
 * and business logic constraints across the platform.
 * 
 * Features:
 * - Type-safe validation with custom decorators
 * - Async validation support
 * - Business rule validation
 * - Schema validation for complex objects
 * - Performance-optimized validation pipelines
 */

import { 
  IsString, 
  IsNumber, 
  IsBoolean, 
  IsArray, 
  IsOptional, 
  IsEmail,
  IsUUID,
  IsISO8601,
  Min, 
  Max, 
  Length,
  IsHexadecimal,
  IsPositive,
  ValidateNested,
  IsEnum,
  ArrayMinSize,
  ArrayMaxSize,
  validate as classValidate,
  ValidationError,
  registerDecorator,
  ValidationOptions,
  ValidatorConstraint,
  ValidatorConstraintInterface,
  ValidationArguments
} from 'class-validator';
import { Transform } from 'class-transformer';
import * as bcrypt from 'bcryptjs';
import { ethers } from 'ethers';
import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import logger from '../config/logger';

// Custom validation decorators
export interface CustomValidationOptions extends ValidationOptions {
  each?: boolean;
}

// Ethereum address validation
@ValidatorConstraint({ async: false })
export class IsEthereumAddressConstraint implements ValidatorConstraintInterface {
  validate(value: any, args: ValidationArguments) {
    if (typeof value !== 'string') return false;
    return ethers.utils.isAddress(value);
  }

  defaultMessage(args: ValidationArguments) {
    return 'Invalid Ethereum address format';
  }
}

export function IsEthereumAddress(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [],
      validator: IsEthereumAddressConstraint,
    });
  };
}

// Token symbol validation
@ValidatorConstraint({ async: false })
export class IsTokenSymbolConstraint implements ValidatorConstraintInterface {
  validate(value: any, args: ValidationArguments) {
    if (typeof value !== 'string') return false;
    return /^[A-Z]{2,10}$/.test(value);
  }

  defaultMessage(args: ValidationArguments) {
    return 'Token symbol must be 2-10 uppercase letters';
  }
}

export function IsTokenSymbol(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [],
      validator: IsTokenSymbolConstraint,
    });
  };
}

// Big number validation for DeFi amounts
@ValidatorConstraint({ async: false })
export class IsBigNumberStringConstraint implements ValidatorConstraintInterface {
  validate(value: any, args: ValidationArguments) {
    if (typeof value !== 'string') return false;
    try {
      ethers.BigNumber.from(value);
      return true;
    } catch {
      return false;
    }
  }

  defaultMessage(args: ValidationArguments) {
    return 'Invalid big number format';
  }
}

export function IsBigNumberString(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [],
      validator: IsBigNumberStringConstraint,
    });
  };
}

// Percentage validation (0-100)
@ValidatorConstraint({ async: false })
export class IsPercentageConstraint implements ValidatorConstraintInterface {
  validate(value: any, args: ValidationArguments) {
    return typeof value === 'number' && value >= 0 && value <= 100;
  }

  defaultMessage(args: ValidationArguments) {
    return 'Percentage must be between 0 and 100';
  }
}

export function IsPercentage(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [],
      validator: IsPercentageConstraint,
    });
  };
}

// Allocation validation (sum equals 100%)
@ValidatorConstraint({ async: false })
export class IsValidAllocationsConstraint implements ValidatorConstraintInterface {
  validate(value: any, args: ValidationArguments) {
    if (!Array.isArray(value)) return false;
    
    // Check if each allocation has required fields
    for (const allocation of value) {
      if (!allocation.protocol || typeof allocation.percentage !== 'number') {
        return false;
      }
    }
    
    // Check if total percentage equals 100
    const totalPercentage = value.reduce((sum, allocation) => sum + allocation.percentage, 0);
    return Math.abs(totalPercentage - 100) < 0.001; // Allow small floating point differences
  }

  defaultMessage(args: ValidationArguments) {
    return 'Total allocation percentage must equal 100%';
  }
}

export function IsValidAllocations(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [],
      validator: IsValidAllocationsConstraint,
    });
  };
}

// Risk tolerance validation
export enum RiskTolerance {
  CONSERVATIVE = 'conservative',
  MODERATE = 'moderate',
  AGGRESSIVE = 'aggressive'
}

// Investment strategy validation
export enum InvestmentStrategy {
  DCA = 'dca', // Dollar Cost Averaging
  YIELD_FARMING = 'yield_farming',
  LIQUIDITY_MINING = 'liquidity_mining',
  ARBITRAGE = 'arbitrage',
  MARKET_NEUTRAL = 'market_neutral'
}

// Protocol validation
export enum SupportedProtocol {
  UNISWAP = 'uniswap',
  SUSHISWAP = 'sushiswap',
  AAVE = 'aave',
  COMPOUND = 'compound',
  CURVE = 'curve',
  MAKER = 'maker',
  YEARN = 'yearn'
}

// DTO Classes for API validation
export class AllocationDto {
  @IsString()
  @IsEnum(SupportedProtocol)
  protocol: string;

  @IsNumber()
  @IsPercentage()
  percentage: number;

  @IsOptional()
  @IsString()
  pool?: string;

  @IsOptional()
  @IsString()
  vault?: string;
}

export class CreatePortfolioDto {
  @IsString()
  @Length(1, 100)
  name: string;

  @IsOptional()
  @IsString()
  @Length(0, 500)
  description?: string;

  @IsString()
  @IsEnum(RiskTolerance)
  riskTolerance: RiskTolerance;

  @IsString()
  @IsEnum(InvestmentStrategy)
  strategy: InvestmentStrategy;

  @IsArray()
  @ArrayMinSize(1)
  @ArrayMaxSize(20)
  @ValidateNested({ each: true })
  @IsValidAllocations()
  allocations: AllocationDto[];

  @IsNumber()
  @IsPositive()
  initialAmount: number;

  @IsOptional()
  @IsBoolean()
  autoRebalance?: boolean;

  @IsOptional()
  @IsNumber()
  @Min(0.01)
  @Max(0.5)
  rebalanceThreshold?: number;
}

export class UpdatePortfolioDto {
  @IsOptional()
  @IsString()
  @Length(1, 100)
  name?: string;

  @IsOptional()
  @IsString()
  @Length(0, 500)
  description?: string;

  @IsOptional()
  @IsString()
  @IsEnum(RiskTolerance)
  riskTolerance?: RiskTolerance;

  @IsOptional()
  @IsArray()
  @ArrayMinSize(1)
  @ArrayMaxSize(20)
  @ValidateNested({ each: true })
  @IsValidAllocations()
  allocations?: AllocationDto[];

  @IsOptional()
  @IsBoolean()
  autoRebalance?: boolean;

  @IsOptional()
  @IsNumber()
  @Min(0.01)
  @Max(0.5)
  rebalanceThreshold?: number;
}

export class DepositDto {
  @IsString()
  @IsUUID()
  portfolioId: string;

  @IsBigNumberString()
  amount: string;

  @IsString()
  @IsTokenSymbol()
  token: string;

  @IsOptional()
  @IsString()
  @IsHexadecimal()
  txHash?: string;
}

export class WithdrawDto {
  @IsString()
  @IsUUID()
  portfolioId: string;

  @IsBigNumberString()
  amount: string;

  @IsString()
  @IsTokenSymbol()
  token: string;

  @IsString()
  @IsEthereumAddress()
  recipient: string;
}

export class RebalanceDto {
  @IsString()
  @IsUUID()
  portfolioId: string;

  @IsOptional()
  @IsArray()
  @ValidateNested({ each: true })
  newAllocations?: AllocationDto[];

  @IsOptional()
  @IsBoolean()
  force?: boolean;
}

export class UserRegistrationDto {
  @IsEmail()
  email: string;

  @IsString()
  @Length(8, 128)
  password: string;

  @IsString()
  @Length(2, 50)
  firstName: string;

  @IsString()
  @Length(2, 50)
  lastName: string;

  @IsOptional()
  @IsString()
  @IsEthereumAddress()
  walletAddress?: string;

  @IsOptional()
  @IsBoolean()
  acceptTerms?: boolean;

  @IsOptional()
  @IsBoolean()
  acceptPrivacyPolicy?: boolean;
}

export class UserLoginDto {
  @IsEmail()
  email: string;

  @IsString()
  password: string;

  @IsOptional()
  @IsBoolean()
  rememberMe?: boolean;
}

export class UserUpdateDto {
  @IsOptional()
  @IsString()
  @Length(2, 50)
  firstName?: string;

  @IsOptional()
  @IsString()
  @Length(2, 50)
  lastName?: string;

  @IsOptional()
  @IsString()
  @IsEthereumAddress()
  walletAddress?: string;

  @IsOptional()
  @IsString()
  @IsEnum(RiskTolerance)
  defaultRiskTolerance?: RiskTolerance;
}

export class PasswordChangeDto {
  @IsString()
  currentPassword: string;

  @IsString()
  @Length(8, 128)
  newPassword: string;
}

export class TransactionDto {
  @IsString()
  @IsUUID()
  portfolioId: string;

  @IsString()
  @IsEnum(['deposit', 'withdraw', 'rebalance', 'swap'])
  type: string;

  @IsBigNumberString()
  amount: string;

  @IsString()
  @IsTokenSymbol()
  token: string;

  @IsOptional()
  @IsString()
  @IsHexadecimal()
  txHash?: string;

  @IsOptional()
  @IsNumber()
  gasUsed?: number;

  @IsOptional()
  @IsBigNumberString()
  gasFee?: string;
}

export class SocialSentimentDto {
  @IsString()
  @IsEnum(SupportedProtocol)
  protocol: string;

  @IsNumber()
  @Min(-1)
  @Max(1)
  sentiment: number;

  @IsNumber()
  @Min(0)
  @Max(1)
  confidence: number;

  @IsNumber()
  @IsPositive()
  mentions: number;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  keywords?: string[];
}

export class CreateAlertDto {
  @IsString()
  @Length(1, 100)
  name: string;

  @IsString()
  @IsEnum(['portfolio_loss', 'protocol_risk', 'social_sentiment', 'price_change'])
  type: string;

  @IsString()
  @IsUUID()
  portfolioId: string;

  @IsString()
  condition: string; // JSON string representing the condition

  @IsOptional()
  @IsBoolean()
  enabled?: boolean;

  @IsOptional()
  @IsString()
  @IsEmail()
  notificationEmail?: string;
}

// Validation middleware for Express
export function validateDto(dtoClass: any) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const dto = new dtoClass();
    Object.assign(dto, req.body);

    try {
      const errors = await classValidate(dto);
      
      if (errors.length > 0) {
        const formattedErrors = formatValidationErrors(errors);
        return res.status(400).json({
          success: false,
          message: 'Validation failed',
          errors: formattedErrors
        });
      }

      req.validatedData = dto;
      next();
    } catch (error) {
      logger.error('Validation error:', error);
      res.status(500).json({
        success: false,
        message: 'Internal validation error'
      });
    }
  };
}

// Format validation errors for API response
function formatValidationErrors(errors: ValidationError[]): any[] {
  const formattedErrors: any[] = [];

  for (const error of errors) {
    const constraints = error.constraints;
    if (constraints) {
      for (const key in constraints) {
        formattedErrors.push({
          field: error.property,
          message: constraints[key],
          value: error.value
        });
      }
    }

    // Handle nested errors
    if (error.children && error.children.length > 0) {
      const nestedErrors = formatValidationErrors(error.children);
      formattedErrors.push(...nestedErrors.map(e => ({
        ...e,
        field: `${error.property}.${e.field}`
      })));
    }
  }

  return formattedErrors;
}

// Business logic validation functions
export class BusinessValidation {
  /**
   * Validate portfolio allocations against risk tolerance
   */
  static validateAllocationsAgainstRisk(
    allocations: AllocationDto[],
    riskTolerance: RiskTolerance
  ): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Define risk limits for each risk tolerance
    const riskLimits: Record<RiskTolerance, { maxSingleAllocation: number; maxVolatileProtocols: number }> = {
      [RiskTolerance.CONSERVATIVE]: { maxSingleAllocation: 30, maxVolatileProtocols: 20 },
      [RiskTolerance.MODERATE]: { maxSingleAllocation: 50, maxVolatileProtocols: 40 },
      [RiskTolerance.AGGRESSIVE]: { maxSingleAllocation: 80, maxVolatileProtocols: 100 }
    };

    const limits = riskLimits[riskTolerance];

    // Check single allocation limits
    for (const allocation of allocations) {
      if (allocation.percentage > limits.maxSingleAllocation) {
        errors.push(
          `Allocation of ${allocation.percentage}% to ${allocation.protocol} exceeds ${limits.maxSingleAllocation}% limit for ${riskTolerance} risk tolerance`
        );
      }
    }

    // Check volatile protocols
    const volatileProtocols = [SupportedProtocol.CURVE, SupportedProtocol.YEARN];
    const volatileAllocation = allocations
      .filter(a => volatileProtocols.includes(a.protocol as SupportedProtocol))
      .reduce((sum, a) => sum + a.percentage, 0);

    if (volatileAllocation > limits.maxVolatileProtocols) {
      errors.push(
        `Total allocation to volatile protocols (${volatileAllocation}%) exceeds ${limits.maxVolatileProtocols}% limit for ${riskTolerance} risk tolerance`
      );
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Validate minimum deposit amount for strategy
   */
  static validateMinimumDeposit(
    amount: number,
    strategy: InvestmentStrategy
  ): { valid: boolean; errors: string[] } {
    const minAmounts: Record<InvestmentStrategy, number> = {
      [InvestmentStrategy.DCA]: 100,
      [InvestmentStrategy.YIELD_FARMING]: 1000,
      [InvestmentStrategy.LIQUIDITY_MINING]: 1000,
      [InvestmentStrategy.ARBITRAGE]: 5000,
      [InvestmentStrategy.MARKET_NEUTRAL]: 10000
    };

    const minRequired = minAmounts[strategy];
    
    if (amount < minRequired) {
      return {
        valid: false,
        errors: [`Minimum deposit for ${strategy} strategy is $${minRequired}`]
      };
    }

    return { valid: true, errors: [] };
  }

  /**
   * Validate protocol compatibility with strategy
   */
  static validateProtocolCompatibility(
    protocols: string[],
    strategy: InvestmentStrategy
  ): { valid: boolean; errors: string[] } {
    const compatibleProtocols: Record<InvestmentStrategy, string[]> = {
      [InvestmentStrategy.DCA]: Object.values(SupportedProtocol),
      [InvestmentStrategy.YIELD_FARMING]: [
        SupportedProtocol.AAVE,
        SupportedProtocol.COMPOUND,
        SupportedProtocol.YEARN
      ],
      [InvestmentStrategy.LIQUIDITY_MINING]: [
        SupportedProtocol.UNISWAP,
        SupportedProtocol.SUSHISWAP,
        SupportedProtocol.CURVE
      ],
      [InvestmentStrategy.ARBITRAGE]: Object.values(SupportedProtocol),
      [InvestmentStrategy.MARKET_NEUTRAL]: [
        SupportedProtocol.UNISWAP,
        SupportedProtocol.AAVE,
        SupportedProtocol.COMPOUND
      ]
    };

    const allowed = compatibleProtocols[strategy];
    const invalid = protocols.filter(p => !allowed.includes(p));

    if (invalid.length > 0) {
      return {
        valid: false,
        errors: [`Protocols ${invalid.join(', ')} are not compatible with ${strategy} strategy`]
      };
    }

    return { valid: true, errors: [] };
  }

  /**
   * Validate rebalance conditions
   */
  static validateRebalanceConditions(
    currentAllocations: Record<string, number>,
    targetAllocations: AllocationDto[],
    threshold: number = 0.05
  ): { shouldRebalance: boolean; reasons: string[] } {
    const reasons: string[] = [];
    let maxDrift = 0;

    for (const target of targetAllocations) {
      const current = currentAllocations[target.protocol] || 0;
      const drift = Math.abs(current - target.percentage) / 100;
      
      if (drift > maxDrift) {
        maxDrift = drift;
      }
      
      if (drift > threshold) {
        reasons.push(
          `${target.protocol} allocation drifted ${(drift * 100).toFixed(2)}% from target`
        );
      }
    }

    return {
      shouldRebalance: maxDrift > threshold,
      reasons
    };
  }
}

// Sanitization functions
export class DataSanitization {
  /**
   * Sanitize string input to prevent XSS
   */
  static sanitizeString(input: string): string {
    if (typeof input !== 'string') return '';
    
    return input
      .replace(/[<>]/g, '') // Remove angle brackets
      .replace(/javascript:/gi, '') // Remove javascript protocol
      .replace(/on\w+=/gi, '') // Remove event handlers
      .replace(/script/gi, '') // Remove script tags
      .trim();
  }

  /**
   * Sanitize email input
   */
  static sanitizeEmail(email: string): string {
    if (typeof email !== 'string') return '';
    
    return email
      .toLowerCase()
      .replace(/[^a-z0-9@._-]/g, '')
      .trim();
  }

  /**
   * Validate and normalize Ethereum address
   */
  static normalizeEthereumAddress(address: string): string | null {
    try {
      return ethers.utils.getAddress(address);
    } catch {
      return null;
    }
  }

  /**
   * Sanitize numeric input
   */
  static sanitizeNumber(
    input: any,
    options?: { min?: number; max?: number; decimals?: number }
  ): number | null {
    const num = parseFloat(input);
    
    if (isNaN(num)) return null;
    
    let result = num;
    
    if (options?.min !== undefined && result < options.min) result = options.min;
    if (options?.max !== undefined && result > options.max) result = options.max;
    if (options?.decimals !== undefined) {
      result = parseFloat(result.toFixed(options.decimals));
    }
    
    return result;
  }

  /**
   * Sanitize object by removing dangerous properties
   */
  static sanitizeObject<T extends Record<string, any>>(
    obj: T,
    allowedFields?: string[]
  ): Partial<T> {
    const sanitized: Partial<T> = {};
    const dangerousFields = ['__proto__', 'constructor', 'prototype'];
    
    for (const [key, value] of Object.entries(obj)) {
      if (dangerousFields.includes(key)) continue;
      if (allowedFields && !allowedFields.includes(key)) continue;
      
      if (typeof value === 'string') {
        sanitized[key as keyof T] = this.sanitizeString(value) as T[keyof T];
      } else if (typeof value === 'number') {
        sanitized[key as keyof T] = value;
      } else if (typeof value === 'boolean') {
        sanitized[key as keyof T] = value;
      } else if (Array.isArray(value)) {
        sanitized[key as keyof T] = value.map(item => 
          typeof item === 'string' ? this.sanitizeString(item) : item
        ) as T[keyof T];
      } else if (value && typeof value === 'object') {
        sanitized[key as keyof T] = this.sanitizeObject(value) as T[keyof T];
      }
    }
    
    return sanitized;
  }
}

// Rate limiting validation
export class RateLimitValidation {
  private static requestCounts = new Map<string, { count: number; resetTime: number }>();
  
  /**
   * Check if request is within rate limit
   */
  static checkRateLimit(
    identifier: string,
    limit: number,
    windowMs: number = 60000
  ): { allowed: boolean; remaining: number; resetTime: number } {
    const now = Date.now();
    const key = `${identifier}:${Math.floor(now / windowMs)}`;
    
    const current = this.requestCounts.get(key);
    
    if (!current) {
      this.requestCounts.set(key, { count: 1, resetTime: now + windowMs });
      return { allowed: true, remaining: limit - 1, resetTime: now + windowMs };
    }
    
    if (current.count >= limit) {
      return { allowed: false, remaining: 0, resetTime: current.resetTime };
    }
    
    current.count++;
    return { allowed: true, remaining: limit - current.count, resetTime: current.resetTime };
  }
  
  /**
   * Clean up old rate limit entries
   */
  static cleanup() {
    const now = Date.now();
    for (const [key, data] of this.requestCounts) {
      if (now > data.resetTime) {
        this.requestCounts.delete(key);
      }
    }
  }
}

// Password validation utilities
export class PasswordValidation {
  /**
   * Validate password strength
   */
  static validatePasswordStrength(password: string): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    if (password.length < 8) {
      errors.push('Password must be at least 8 characters long');
    }
    
    if (!/[A-Z]/.test(password)) {
      errors.push('Password must contain at least one uppercase letter');
    }
    
    if (!/[a-z]/.test(password)) {
      errors.push('Password must contain at least one lowercase letter');
    }
    
    if (!/\d/.test(password)) {
      errors.push('Password must contain at least one number');
    }
    
    if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
      errors.push('Password must contain at least one special character');
    }
    
    // Check for common patterns
    if (/(.)\1{2,}/.test(password)) {
      errors.push('Password cannot contain repeated characters');
    }
    
    if (/123456|password|qwerty/i.test(password)) {
      errors.push('Password is too common');
    }
    
    return { valid: errors.length === 0, errors };
  }
  
  /**
   * Hash password with bcrypt
   */
  static async hashPassword(password: string): Promise<string> {
    const saltRounds = 12;
    return bcrypt.hash(password, saltRounds);
  }
  
  /**
   * Verify password against hash
   */
  static async verifyPassword(password: string, hash: string): Promise<boolean> {
    return bcrypt.compare(password, hash);
  }
}

// Input normalization
export class InputNormalization {
  /**
   * Normalize protocol allocations to ensure they sum to 100%
   */
  static normalizeAllocations(allocations: AllocationDto[]): AllocationDto[] {
    const total = allocations.reduce((sum, allocation) => sum + allocation.percentage, 0);
    
    if (total === 0) return allocations;
    
    return allocations.map(allocation => ({
      ...allocation,
      percentage: (allocation.percentage / total) * 100
    }));
  }
  
  /**
   * Normalize token amounts to consistent precision
   */
  static normalizeTokenAmount(amount: string, decimals: number = 18): string {
    try {
      const bn = ethers.BigNumber.from(amount);
      const divisor = ethers.BigNumber.from(10).pow(decimals);
      const normalized = bn.div(divisor).mul(divisor);
      return normalized.toString();
    } catch {
      return '0';
    }
  }
}

// Export types and interfaces
export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface SanitizationOptions {
  allowedFields?: string[];
  maxLength?: number;
  allowHtml?: boolean;
}

export {
  RiskTolerance,
  InvestmentStrategy,
  SupportedProtocol
};

// Default export for convenience
export default {
  validateDto,
  BusinessValidation,
  DataSanitization,
  RateLimitValidation,
  PasswordValidation,
  InputNormalization,
  formatValidationErrors
};