/**
 * Agent Routes
 * 
 * This module defines the API routes for AI trading agents, including endpoints for
 * agent creation, management, and interaction. These routes handle requests related to
 * agent operations, trading signals, and performance monitoring.
 * 
 * @module AgentRoutes
 * @author Minos-AI Team
 * @date January 8, 2025
 */

import { Router } from 'express';
import { Controller, Get, Post, Put, Delete, Body, Param, Query, UseGuards, Req } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiQuery, ApiBody } from '@nestjs/swagger';
import { JwtAuthGuard } from '../guards/jwt-auth.guard';
import { RolesGuard } from '../guards/roles.guard';
import { Roles } from '../decorators/roles.decorator';
import { AgentService } from '../services/agent.service';
import { 
  CreateAgentDto, 
  UpdateAgentConfigDto, 
  AgentPredictionDto,
  TradeSignalDto,
  AgentPerformanceDto
} from '../dtos/agent.dto';

@ApiTags('agents')
@Controller('agents')
@UseGuards(JwtAuthGuard, RolesGuard)
export class AgentController {
  constructor(private readonly agentService: AgentService) {}

  /**
   * Get all available agent models
   */
  @Get('models')
  @ApiOperation({ summary: 'Get all available AI agent models' })
  @ApiResponse({ status: 200, description: 'List of available agent models' })
  async getAvailableAgentModels() {
    return this.agentService.getAvailableAgentModels();
  }

  /**
   * Get all agents for the authenticated user
   */
  @Get()
  @ApiOperation({ summary: 'Get all agents for the authenticated user' })
  @ApiResponse({ status: 200, description: 'List of user agents' })
  async getAgents(@Req() req) {
    const userId = req.user.id;
    return this.agentService.getAgentsByUser(userId);
  }

  /**
   * Get a specific agent by ID
   */
  @Get(':id')
  @ApiOperation({ summary: 'Get a specific agent by ID' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Agent details' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getAgent(@Param('id') agentId: string) {
    const agent = await this.agentService.getAgentById(agentId);
    if (!agent) {
      throw new Error('Agent not found');
    }
    return agent;
  }

  /**
   * Create a new agent
   */
  @Post()
  @ApiOperation({ summary: 'Create a new AI trading agent' })
  @ApiBody({ type: CreateAgentDto })
  @ApiResponse({ status: 201, description: 'Agent created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async createAgent(@Req() req, @Body() createAgentDto: CreateAgentDto) {
    const owner = req.user;
    return this.agentService.createAgent(owner, createAgentDto);
  }

  /**
   * Confirm agent creation after transaction is processed
   */
  @Post(':id/confirm')
  @ApiOperation({ summary: 'Confirm agent creation after transaction is processed' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ schema: { 
    type: 'object', 
    properties: { 
      transactionSignature: { type: 'string' } 
    } 
  }})
  @ApiResponse({ status: 200, description: 'Agent confirmed successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  @ApiResponse({ status: 400, description: 'Invalid transaction' })
  async confirmAgentCreation(
    @Param('id') agentId: string,
    @Body('transactionSignature') transactionSignature: string
  ) {
    return this.agentService.confirmAgentCreation(agentId, transactionSignature);
  }

  /**
   * Update agent configuration
   */
  @Put(':id')
  @ApiOperation({ summary: 'Update agent configuration' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ type: UpdateAgentConfigDto })
  @ApiResponse({ status: 200, description: 'Agent updated successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async updateAgent(
    @Param('id') agentId: string,
    @Body() updateAgentDto: UpdateAgentConfigDto
  ) {
    return this.agentService.updateAgentConfig(agentId, updateAgentDto);
  }

  /**
   * Deactivate an agent
   */
  @Delete(':id')
  @ApiOperation({ summary: 'Deactivate an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Agent deactivated successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async deactivateAgent(@Param('id') agentId: string) {
    return this.agentService.deactivateAgent(agentId);
  }

  /**
   * Generate trading predictions
   */
  @Post(':id/predict')
  @ApiOperation({ summary: 'Generate trading predictions with an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ type: AgentPredictionDto })
  @ApiResponse({ status: 200, description: 'Predictions generated successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async generatePredictions(
    @Param('id') agentId: string,
    @Body() predictionDto: AgentPredictionDto
  ) {
    return this.agentService.generatePredictions(
      agentId,
      predictionDto.market,
      predictionDto.timeframe
    );
  }

  /**
   * Execute a trade
   */
  @Post(':id/trade')
  @ApiOperation({ summary: 'Execute a trade based on agent signals' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ type: TradeSignalDto })
  @ApiResponse({ status: 200, description: 'Trade executed successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  @ApiResponse({ status: 400, description: 'Invalid trade parameters' })
  async executeTrade(
    @Param('id') agentId: string,
    @Body() tradeSignalDto: TradeSignalDto
  ) {
    return this.agentService.executeTrade(
      agentId,
      tradeSignalDto.vaultId,
      tradeSignalDto
    );
  }

  /**
   * Update agent performance
   */
  @Put(':id/performance')
  @ApiOperation({ summary: 'Update agent performance metrics' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ type: AgentPerformanceDto })
  @ApiResponse({ status: 200, description: 'Performance updated successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async updatePerformance(
    @Param('id') agentId: string,
    @Body() performanceDto: AgentPerformanceDto
  ) {
    return this.agentService.updateAgentPerformance(agentId, performanceDto);
  }

  /**
   * Link an agent to a vault
   */
  @Post(':id/link-vault')
  @ApiOperation({ summary: 'Link an agent to a vault' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ schema: { 
    type: 'object', 
    properties: { 
      vaultId: { type: 'string' },
      allocation: { type: 'number' }
    },
    required: ['vaultId', 'allocation']
  }})
  @ApiResponse({ status: 200, description: 'Agent linked to vault successfully' })
  @ApiResponse({ status: 404, description: 'Agent or vault not found' })
  @ApiResponse({ status: 400, description: 'Invalid parameters' })
  async linkVault(
    @Param('id') agentId: string,
    @Body('vaultId') vaultId: string,
    @Body('allocation') allocation: number
  ) {
    return this.agentService.linkAgentToVault(agentId, vaultId, allocation);
  }

  /**
   * Get agent historical performance
   */
  @Get(':id/performance')
  @ApiOperation({ summary: 'Get agent historical performance' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiQuery({ name: 'startDate', required: false, description: 'Start date (ISO format)' })
  @ApiQuery({ name: 'endDate', required: false, description: 'End date (ISO format)' })
  @ApiResponse({ status: 200, description: 'Historical performance data' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getPerformance(
    @Param('id') agentId: string,
    @Query('startDate') startDateStr?: string,
    @Query('endDate') endDateStr?: string
  ) {
    const startDate = startDateStr ? new Date(startDateStr) : undefined;
    const endDate = endDateStr ? new Date(endDateStr) : undefined;
    
    return this.agentService.getAgentHistoricalPerformance(agentId, startDate, endDate);
  }

  /**
   * Get agent active positions
   */
  @Get(':id/positions')
  @ApiOperation({ summary: 'Get agent active trading positions' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Active positions' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getPositions(@Param('id') agentId: string) {
    return this.agentService.getAgentPositions(agentId);
  }

  /**
   * Get vaults linked to agent
   */
  @Get(':id/vaults')
  @ApiOperation({ summary: 'Get vaults linked to an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Linked vaults' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getLinkedVaults(@Param('id') agentId: string) {
    return this.agentService.getLinkedVaults(agentId);
  }

  /**
   * Reset agent
   */
  @Post(':id/reset')
  @ApiOperation({ summary: "Reset agent's allocations and trade history" })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Agent reset successfully' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async resetAgent(@Param('id') agentId: string) {
    return this.agentService.resetAgent(agentId);
  }

  /**
   * Get market recommendations
   */
  @Get(':id/recommendations')
  @ApiOperation({ summary: 'Get market recommendations from an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiResponse({ status: 200, description: 'Market recommendations' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getRecommendations(@Param('id') agentId: string) {
    return this.agentService.getMarketRecommendations(agentId);
  }

  /**
   * Get optimized strategy parameters
   */
  @Get(':id/optimize')
  @ApiOperation({ summary: 'Get optimized strategy parameters for an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiQuery({ name: 'market', required: true, description: 'Market to optimize for' })
  @ApiResponse({ status: 200, description: 'Optimized strategy parameters' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getOptimizedStrategy(
    @Param('id') agentId: string,
    @Query('market') market: string
  ) {
    return this.agentService.getOptimizedStrategyParams(agentId, market);
  }

  /**
   * Backtest agent strategy
   */
  @Post(':id/backtest')
  @ApiOperation({ summary: "Backtest agent's strategy on historical data" })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ schema: { 
    type: 'object', 
    properties: { 
      market: { type: 'string' },
      startDate: { type: 'string', format: 'date-time' },
      endDate: { type: 'string', format: 'date-time' }
    },
    required: ['market', 'startDate', 'endDate']
  }})
  @ApiResponse({ status: 200, description: 'Backtest results' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async backtestStrategy(
    @Param('id') agentId: string,
    @Body('market') market: string,
    @Body('startDate') startDate: Date,
    @Body('endDate') endDate: Date
  ) {
    return this.agentService.backtestAgentStrategy(agentId, market, startDate, endDate);
  }

  /**
   * Calculate trade risk
   */
  @Post(':id/risk')
  @ApiOperation({ summary: "Calculate risk metrics for agent's proposed trade" })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiBody({ schema: { 
    type: 'object', 
    properties: { 
      market: { type: 'string' },
      direction: { type: 'string', enum: ['buy', 'sell'] },
      amount: { type: 'number' }
    },
    required: ['market', 'direction', 'amount']
  }})
  @ApiResponse({ status: 200, description: 'Risk assessment' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async calculateRisk(
    @Param('id') agentId: string,
    @Body('market') market: string,
    @Body('direction') direction: 'buy' | 'sell',
    @Body('amount') amount: number
  ) {
    return this.agentService.calculateTradeRisk(agentId, market, direction, amount);
  }

  /**
   * Get detailed market analysis
   */
  @Get(':id/analyze')
  @ApiOperation({ summary: 'Get detailed market analysis from an agent' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiQuery({ name: 'market', required: true, description: 'Market to analyze' })
  @ApiQuery({ name: 'timeframe', required: false, description: 'Timeframe for analysis' })
  @ApiResponse({ status: 200, description: 'Detailed market analysis' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async getMarketAnalysis(
    @Param('id') agentId: string,
    @Query('market') market: string,
    @Query('timeframe') timeframe?: string
  ) {
    return this.agentService.getDetailedMarketAnalysis(agentId, market, timeframe);
  }

  /**
   * Export agent data
   */
  @Get(':id/export')
  @ApiOperation({ summary: 'Export agent configuration and history' })
  @ApiParam({ name: 'id', description: 'Agent ID' })
  @ApiQuery({ name: 'format', required: false, enum: ['json', 'csv'], description: 'Export format' })
  @ApiResponse({ status: 200, description: 'Agent data export' })
  @ApiResponse({ status: 404, description: 'Agent not found' })
  async exportAgentData(
    @Param('id') agentId: string,
    @Query('format') format?: 'json' | 'csv'
  ) {
    return this.agentService.exportAgentData(agentId, format);
  }
}

// NestJS module definition
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [ConfigModule],
  controllers: [AgentController],
  providers: [AgentService],
  exports: [AgentService]
})
export class AgentModule {}