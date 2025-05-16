# Optimize portfolio
            if self.config.use_portfolio_optimization:
                strategy = await self._optimize_strategy(strategy, request)
            
            # Validate strategy
            is_valid, validation_errors = await self.strategy_validator.validate_strategy(
                strategy, request.market_state
            )
            
            if not is_valid:
                logger.warning(f"Strategy validation failed: {validation_errors}")
                # Apply fallback strategy
                strategy = await self._apply_fallback_strategy(strategy, validation_errors)
            
            # Calculate risk assessment
            risk_assessment = await self._assess_risk(strategy, request.market_state)
            
            # Create response
            processing_time = time.time() - start_time
            response = InferenceResponse(
                request_id=request.request_id,
                strategy=strategy,
                metadata={
                    'model_version': self.model_manager.model_version,
                    'processing_time': processing_time,
                    'validation_errors': validation_errors,
                    'cache_hit': False,
                    'social_data_weight': self.config.social_data_weight
                },
                processing_time=processing_time,
                confidence=strategy.confidence,
                risk_assessment=risk_assessment,
                model_version=self.model_manager.model_version
            )
            
            # Cache strategy if enabled
            if self.config.cache_strategies:
                await self._cache_strategy(request, response)
            
            # Log prediction if enabled
            if self.config.log_predictions:
                await self._log_prediction(request, response)
            
            # Update metrics
            PORTFOLIO_VALUE.set(sum(strategy.allocations.values()))
            RISK_SCORE.set(strategy.risk_score)
            CONFIDENCE_SCORE.set(strategy.confidence)
            
            # Record processing stats
            self.processing_stats['inference_time'].append(processing_time)
            self.processing_stats['confidence'].append(strategy.confidence)
            self.processing_stats['risk_score'].append(strategy.risk_score)
            
            return response
            
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return error response
            error_response = InferenceResponse(
                request_id=request.request_id,
                strategy=None,
                metadata={'error': str(e)},
                processing_time=time.time() - start_time,
                confidence=0.0,
                risk_assessment={},
                model_version=self.model_manager.model_version
            )
            
            return error_response
    
    async def _get_cached_strategy(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Retrieve cached strategy if available"""
        cache_key = self._create_cache_key(request)
        
        # Check in-memory cache first
        if cache_key in self.strategy_cache:
            cached_data = self.strategy_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.config.strategy_ttl:
                cached_data['response'].metadata['cache_hit'] = True
                return cached_data['response']
            else:
                # Remove expired entry
                del self.strategy_cache[cache_key]
        
        # Check Redis cache if available
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    response = pickle.loads(cached_data)
                    response.metadata['cache_hit'] = True
                    return response
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        return None
    
    async def _cache_strategy(self, request: InferenceRequest, response: InferenceResponse):
        """Cache strategy for future requests"""
        cache_key = self._create_cache_key(request)
        
        # Cache in memory
        self.strategy_cache[cache_key] = {
            'timestamp': time.time(),
            'response': response
        }
        
        # Cache in Redis if available
        if self.redis_client:
            try:
                serialized_response = pickle.dumps(response)
                self.redis_client.setex(cache_key, self.config.strategy_ttl, serialized_response)
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
    
    def _create_cache_key(self, request: InferenceRequest) -> str:
        """Create cache key for request"""
        # Create a hash of the market state and request parameters
        state_hash = hashlib.md5(str(asdict(request.market_state)).encode()).hexdigest()
        return f"strategy:{state_hash}:{request.timestamp.isoformat()}"
    
    async def _update_social_data(self):
        """Update social sentiment data if needed"""
        if not self.social_orchestrator:
            return
        
        # Check if update is needed
        time_since_update = datetime.utcnow() - self.last_social_update
        if time_since_update.total_seconds() < self.config.social_update_interval:
            return
        
        try:
            # This would typically be done in the background
            # For now, we'll just update the timestamp
            self.last_social_update = datetime.utcnow()
            logger.debug("Social data updated")
        except Exception as e:
            logger.warning(f"Failed to update social data: {e}")
    
    async def _predictions_to_strategy(self, predictions: Dict[str, torch.Tensor], market_state: MarketState) -> Strategy:
        """Convert model predictions to Strategy object"""
        # Extract allocations from predictions
        if 'strategy' in predictions:
            strategy_logits = predictions['strategy'].cpu().numpy().flatten()
        else:
            # Fallback to equal allocation
            n_assets = len(market_state.prices)
            strategy_logits = np.ones(n_assets) / n_assets
        
        # Get asset names
        assets = list(market_state.prices.keys())
        
        # Create allocations dictionary
        allocations = {}
        for i, asset in enumerate(assets):
            if i < len(strategy_logits):
                allocations[asset] = float(strategy_logits[i])
        
        # Normalize allocations
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {asset: alloc / total_allocation for asset, alloc in allocations.items()}
        
        # Extract other predictions
        confidence = float(predictions.get('confidence', torch.tensor(0.5)).item())
        risk_score = float(predictions.get('risk', torch.tensor(0.5)).item())
        price_prediction = float(predictions.get('price', torch.tensor(0.0)).item())
        
        # Generate actions based on allocations
        actions = []
        for asset, allocation in allocations.items():
            if allocation > 0.01:  # Minimum threshold
                actions.append({
                    'action': 'allocate',
                    'asset': asset,
                    'allocation': allocation,
                    'predicted_price': price_prediction,
                    'timestamp': datetime.utcnow()
                })
        
        # Create strategy object
        strategy = Strategy(
            timestamp=datetime.utcnow(),
            allocations=allocations,
            actions=actions,
            confidence=confidence,
            risk_score=risk_score,
            expected_return=self._calculate_expected_return(allocations, predictions, market_state),
            max_drawdown=self._estimate_max_drawdown(allocations, risk_score),
            reasoning=self._generate_reasoning(predictions, market_state)
        )
        
        return strategy
    
    async def _apply_social_sentiment(self, strategy: Strategy, market_state: MarketState) -> Strategy:
        """Apply social sentiment adjustment to strategy"""
        if not self.social_orchestrator:
            return strategy
        
        try:
            # Get social sentiment for each protocol
            social_adjustments = {}
            
            for asset in strategy.allocations.keys():
                social_metrics = await self.social_orchestrator.get_protocol_sentiment(asset)
                
                # Calculate sentiment adjustment
                sentiment_score = social_metrics.sentiment_score
                sentiment_confidence = social_metrics.sentiment_confidence
                
                # Apply adjustment based on sentiment strength
                adjustment_factor = 1.0 + (sentiment_score * sentiment_confidence * self.config.social_data_weight)
                social_adjustments[asset] = adjustment_factor
            
            # Apply adjustments to allocations
            adjusted_allocations = {}
            for asset, allocation in strategy.allocations.items():
                adjustment = social_adjustments.get(asset, 1.0)
                adjusted_allocations[asset] = allocation * adjustment
            
            # Normalize adjusted allocations
            total_allocation = sum(adjusted_allocations.values())
            if total_allocation > 0:
                adjusted_allocations = {
                    asset: alloc / total_allocation 
                    for asset, alloc in adjusted_allocations.items()
                }
            
            # Update strategy
            strategy.allocations = adjusted_allocations
            
            # Add social reasoning
            strategy.reasoning += f" Social sentiment adjustment applied with weight {self.config.social_data_weight}."
            
        except Exception as e:
            logger.warning(f"Failed to apply social sentiment: {e}")
        
        return strategy
    
    async def _optimize_strategy(self, strategy: Strategy, request: InferenceRequest) -> Strategy:
        """Optimize strategy using portfolio optimization techniques"""
        try:
            # Prepare optimization inputs
            expected_returns = {
                asset: strategy.expected_return * allocation  # Simplified
                for asset, allocation in strategy.allocations.items()
            }
            
            risk_estimates = {}
            for asset in strategy.allocations.keys():
                if asset in request.market_state.volatility:
                    risk_estimates[asset] = request.market_state.volatility[asset]
                else:
                    risk_estimates[asset] = 0.02  # Default risk estimate
            
            # Get current portfolio if available
            current_portfolio = request.portfolio_state
            
            # Optimize allocations
            optimized_allocations = await self.portfolio_optimizer.optimize_portfolio(
                strategy.allocations,
                expected_returns,
                risk_estimates,
                current_portfolio
            )
            
            # Update strategy with optimized allocations
            strategy.allocations = optimized_allocations
            
            # Recalculate expected return and risk
            strategy.expected_return = self._calculate_expected_return(
                optimized_allocations, {}, request.market_state
            )
            
            # Add optimization note to reasoning
            strategy.reasoning += f" Portfolio optimized using {self.config.optimization_method} method."
            
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")
        
        return strategy
    
    async def _apply_fallback_strategy(self, strategy: Strategy, errors: List[str]) -> Strategy:
        """Apply fallback strategy when validation fails"""
        logger.info("Applying fallback strategy due to validation errors")
        
        # Create a conservative fallback strategy
        if not strategy or not strategy.allocations:
            # Default to equal weighted portfolio
            assets = list(strategy.allocations.keys()) if strategy else ['USDC']
            n_assets = len(assets)
            fallback_allocations = {asset: 1.0 / n_assets for asset in assets}
        else:
            # Apply conservative adjustments
            fallback_allocations = {}
            total_allocation = sum(strategy.allocations.values())
            
            for asset, allocation in strategy.allocations.items():
                # Cap individual allocations
                capped_allocation = min(allocation, 0.2)  # Max 20% per asset
                fallback_allocations[asset] = capped_allocation
            
            # Add remainder to USDC (cash equivalent)
            remaining = 1.0 - sum(fallback_allocations.values())
            if remaining > 0:
                fallback_allocations['USDC'] = remaining
        
        # Create fallback strategy
        fallback_strategy = Strategy(
            timestamp=datetime.utcnow(),
            allocations=fallback_allocations,
            actions=[],
            confidence=0.3,  # Low confidence for fallback
            risk_score=0.1,  # Conservative risk
            expected_return=0.02,  # Conservative return
            max_drawdown=0.05,
            reasoning=f"Fallback strategy applied due to validation errors: {'; '.join(errors)}"
        )
        
        return fallback_strategy
    
    async def _assess_risk(self, strategy: Strategy, market_state: MarketState) -> Dict[str, float]:
        """Comprehensive risk assessment of the strategy"""
        risk_assessment = {}
        
        # Portfolio concentration risk
        allocations = list(strategy.allocations.values())
        herfindahl_index = sum(alloc ** 2 for alloc in allocations)
        risk_assessment['concentration_risk'] = herfindahl_index
        
        # Asset volatility risk
        weighted_volatility = 0.0
        for asset, allocation in strategy.allocations.items():
            volatility = market_state.volatility.get(asset, 0.02)
            weighted_volatility += allocation * volatility
        risk_assessment['volatility_risk'] = weighted_volatility
        
        # Liquidity risk
        weighted_liquidity_risk = 0.0
        for asset, allocation in strategy.allocations.items():
            liquidity = market_state.liquidity.get(asset, 1.0)
            liquidity_risk = 1.0 - liquidity
            weighted_liquidity_risk += allocation * liquidity_risk
        risk_assessment['liquidity_risk'] = weighted_liquidity_risk
        
        # Social sentiment risk
        if self.social_orchestrator:
            sentiment_risk = 0.0
            for asset in strategy.allocations.keys():
                try:
                    social_metrics = await self.social_orchestrator.get_protocol_sentiment(asset)
                    if social_metrics.risk_signals:
                        sentiment_risk += len(social_metrics.risk_signals) * 0.1
                except Exception:
                    pass
            risk_assessment['sentiment_risk'] = min(sentiment_risk, 1.0)
        
        # Overall risk score (weighted combination)
        overall_risk = (
            risk_assessment.get('concentration_risk', 0) * 0.3 +
            risk_assessment.get('volatility_risk', 0) * 0.35 +
            risk_assessment.get('liquidity_risk', 0) * 0.25 +
            risk_assessment.get('sentiment_risk', 0) * 0.1
        )
        risk_assessment['overall_risk'] = overall_risk
        
        return risk_assessment
    
    def _calculate_expected_return(self, allocations: Dict[str, float], predictions: Dict, market_state: MarketState) -> float:
        """Calculate expected portfolio return"""
        # Simplified expected return calculation
        # In practice, this would use more sophisticated models
        
        if 'price' in predictions:
            # Use model price prediction
            price_change = float(predictions['price'].item())
            return price_change * sum(allocations.values())
        else:
            # Use historical or assumed returns
            total_return = 0.0
            for asset, allocation in allocations.items():
                # Assume base return of 5% annually, adjusted by allocation
                base_return = 0.05
                total_return += allocation * base_return
            return total_return
    
    def _estimate_max_drawdown(self, allocations: Dict[str, float], risk_score: float) -> float:
        """Estimate maximum potential drawdown"""
        # Simplified drawdown estimation
        # Higher allocation in single asset = higher potential drawdown
        max_allocation = max(allocations.values()) if allocations else 0
        concentration_penalty = max_allocation * 0.5
        
        # Risk score penalty
        risk_penalty = risk_score * 0.3
        
        # Combined estimate
        estimated_drawdown = concentration_penalty + risk_penalty
        
        return min(estimated_drawdown, 0.5)  # Cap at 50%
    
    def _generate_reasoning(self, predictions: Dict, market_state: MarketState) -> str:
        """Generate human-readable reasoning for the strategy"""
        reasoning_parts = []
        
        # Model confidence
        if 'confidence' in predictions:
            confidence = float(predictions['confidence'].item())
            if confidence > 0.8:
                reasoning_parts.append("High model confidence in predictions.")
            elif confidence < 0.5:
                reasoning_parts.append("Model confidence is low, strategy is conservative.")
        
        # Market conditions
        avg_volatility = np.mean(list(market_state.volatility.values()))
        if avg_volatility > 0.05:
            reasoning_parts.append("High market volatility detected, applying risk management.")
        
        # Strategy type
        if 'risk' in predictions:
            risk_level = float(predictions['risk'].item())
            if risk_level < 0.3:
                reasoning_parts.append("Low-risk strategy due to market conditions.")
            elif risk_level > 0.7:
                reasoning_parts.append("Aggressive strategy with higher expected returns.")
        
        return " ".join(reasoning_parts) if reasoning_parts else "Strategy generated based on comprehensive market analysis."
    
    async def _log_prediction(self, request: InferenceRequest, response: InferenceResponse):
        """Log prediction for analysis and monitoring"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.request_id,
            'confidence': response.confidence,
            'risk_score': response.strategy.risk_score if response.strategy else 0,
            'processing_time': response.processing_time,
            'allocations': response.strategy.allocations if response.strategy else {},
            'market_state_hash': hashlib.md5(str(request.market_state).encode()).hexdigest()
        }
        
        # Log to file (or database in production)
        logger.info(f"Prediction logged: {json.dumps(log_data)}")
        
        # Send to wandb if configured
        if hasattr(self, 'wandb_run'):
            wandb.log(log_data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference engine performance statistics"""
        stats = {
            'total_requests': len(self.processing_stats.get('inference_time', [])),
            'avg_inference_time': np.mean(self.processing_stats['inference_time']) if self.processing_stats['inference_time'] else 0,
            'avg_confidence': np.mean(self.processing_stats['confidence']) if self.processing_stats['confidence'] else 0,
            'avg_risk_score': np.mean(self.processing_stats['risk_score']) if self.processing_stats['risk_score'] else 0,
            'cache_stats': self.feature_processor.get_cache_stats(),
            'validation_stats': self.strategy_validator.get_validation_stats(),
            'model_info': self.model_manager.get_model_info()
        }
        
        return stats
    
    async def shutdown(self):
        """Shutdown the inference engine gracefully"""
        logger.info("Shutting down Deucalion Inference Engine...")
        
        # Close thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        # Stop social data orchestrator
        if self.social_orchestrator:
            await self.social_orchestrator.stop()
        
        logger.info("Deucalion Inference Engine shutdown complete")


class ABTestManager:
    """Manage A/B tests for strategy variants"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.test_variants = config.test_variants or []
        self.test_results = defaultdict(list)
        self.current_tests = {}
        
    def assign_variant(self, request_id: str) -> str:
        """Assign A/B test variant to a request"""
        if not self.test_variants:
            return 'control'
        
        # Simple hash-based assignment
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        variant_index = hash_value % len(self.test_variants)
        
        return self.test_variants[variant_index]
    
    def record_result(self, request_id: str, variant: str, performance_metric: float):
        """Record A/B test result"""
        self.test_results[variant].append({
            'request_id': request_id,
            'performance': performance_metric,
            'timestamp': datetime.utcnow()
        })
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of A/B test results"""
        summary = {}
        
        for variant, results in self.test_results.items():
            if results:
                performances = [r['performance'] for r in results]
                summary[variant] = {
                    'count': len(results),
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'confidence_interval': np.percentile(performances, [2.5, 97.5]).tolist()
                }
        
        return summary


class BatchInferenceEngine:
    """Optimized engine for batch inference processing"""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.batch_queue = asyncio.Queue()
        self.results_cache = {}
        
    async def process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Process multiple requests in batch for efficiency"""
        batch_size = self.inference_engine.config.batch_size
        responses = []
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_responses = await self._process_single_batch(batch)
            responses.extend(batch_responses)
        
        return responses
    
    async def _process_single_batch(self, batch: List[InferenceRequest]) -> List[InferenceResponse]:
        """Process a single batch of requests"""
        # Prepare features for all requests in batch
        batch_features = []
        for request in batch:
            features = await self.inference_engine.feature_processor.process_market_state(
                request.market_state
            )
            batch_features.append(features)
        
        # Stack features for batch processing
        if batch_features:
            batched_features = torch.cat(batch_features, dim=0)
            
            # Run batch inference
            batch_predictions = await self.inference_engine.model_manager.predict(batched_features)
            
            # Split results back to individual predictions
            responses = []
            for i, request in enumerate(batch):
                # Extract predictions for this request
                single_predictions = {}
                for key, tensor in batch_predictions.items():
                    single_predictions[key] = tensor[i:i+1]
                
                # Convert to strategy
                strategy = await self.inference_engine._predictions_to_strategy(
                    single_predictions, request.market_state
                )
                
                # Create response
                response = InferenceResponse(
                    request_id=request.request_id,
                    strategy=strategy,
                    metadata={'batch_processed': True},
                    processing_time=0.0,  # Will be updated
                    confidence=strategy.confidence,
                    risk_assessment={},
                    model_version=self.inference_engine.model_manager.model_version
                )
                
                responses.append(response)
            
            return responses
        
        return []


# Utility functions and classes
class InferenceMonitor:
    """Monitor inference performance and health"""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.health_checks = []
        self.alert_thresholds = {
            'max_latency': 10.0,  # seconds
            'min_confidence': 0.3,
            'max_error_rate': 0.1
        }
        self.metrics_history = defaultdict(deque)
        
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'alerts': []
        }
        
        # Check model status
        health_status['checks']['model_loaded'] = self.inference_engine.model_manager.current_model is not None
        
        # Check cache performance
        cache_stats = self.inference_engine.feature_processor.get_cache_stats()
        health_status['checks']['cache_hit_rate'] = cache_stats.get('cache_hit_rate', 0)
        
        # Check processing times
        recent_times = list(self.inference_engine.processing_stats.get('inference_time', []))[-100:]
        if recent_times:
            avg_latency = np.mean(recent_times)
            health_status['checks']['avg_latency'] = avg_latency
            
            if avg_latency > self.alert_thresholds['max_latency']:
                health_status['alerts'].append(f"High latency detected: {avg_latency:.2f}s")
                health_status['overall_status'] = 'warning'
        
        # Check confidence scores
        recent_confidence = list(self.inference_engine.processing_stats.get('confidence', []))[-100:]
        if recent_confidence:
            avg_confidence = np.mean(recent_confidence)
            health_status['checks']['avg_confidence'] = avg_confidence
            
            if avg_confidence < self.alert_thresholds['min_confidence']:
                health_status['alerts'].append(f"Low confidence detected: {avg_confidence:.2f}")
                health_status['overall_status'] = 'warning'
        
        # Check system resources
        health_status['checks']['cpu_usage'] = psutil.cpu_percent()
        health_status['checks']['memory_usage'] = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            health_status['checks']['gpu_memory'] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            )
        
        return health_status
    
    def add_custom_check(self, check_name: str, check_function: Callable):
        """Add custom health check"""
        self.health_checks.append((check_name, check_function))
    
    async def get_performance_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # This would typically query from a time-series database
        # For now, we'll use the in-memory stats
        stats = self.inference_engine.get_performance_stats()
        
        # Add time-series data
        metrics = {
            'period_hours': hours_back,
            'summary': stats,
            'trends': {},
            'distributions': {}
        }
        
        # Calculate trends
        if self.inference_engine.processing_stats['inference_time']:
            recent_times = self.inference_engine.processing_stats['inference_time'][-100:]
            old_times = self.inference_engine.processing_stats['inference_time'][-200:-100]
            
            if old_times:
                trend = (np.mean(recent_times) - np.mean(old_times)) / np.mean(old_times)
                metrics['trends']['latency_trend'] = trend
        
        return metrics


# Main API classes for external integration
class InferenceAPI:
    """HTTP API for inference requests"""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.request_counter = 0
        
    async def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for strategy prediction"""
        self.request_counter += 1
        
        try:
            # Convert market data to MarketState
            market_state = self._dict_to_market_state(market_data)
            
            # Create inference request
            request = InferenceRequest(
                timestamp=datetime.utcnow(),
                market_state=market_state,
                request_id=f"api_{self.request_counter}"
            )
            
            # Generate strategy
            response = await self.inference_engine.generate_strategy(request)
            
            # Convert to API response format
            api_response = {
                'request_id': response.request_id,
                'success': response.is_valid,
                'strategy': {
                    'allocations': response.strategy.allocations if response.strategy else {},
                    'confidence': response.confidence,
                    'risk_score': response.strategy.risk_score if response.strategy else 0,
                    'expected_return': response.strategy.expected_return if response.strategy else 0,
                    'reasoning': response.strategy.reasoning if response.strategy else ""
                },
                'metadata': response.metadata,
                'processing_time': response.processing_time
            }
            
            return api_response
            
        except Exception as e:
            logger.error(f"API prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def _dict_to_market_state(self, data: Dict[str, Any]) -> MarketState:
        """Convert dictionary to MarketState object"""
        return MarketState(
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
            prices=data.get('prices', {}),
            volumes=data.get('volumes', {}),
            liquidity=data.get('liquidity', {}),
            volatility=data.get('volatility', {}),
            social_sentiment=data.get('social_sentiment', {}),
            technical_indicators=data.get('technical_indicators', {}),
            protocol_metrics=data.get('protocol_metrics', {}),
            risk_metrics=data.get('risk_metrics', {})
        )


# Export main classes and functions
__all__ = [
    'InferenceConfig',
    'InferenceRequest',
    'InferenceResponse',
    'FeatureProcessor',
    'ModelManager',
    'PortfolioOptimizer',
    'StrategyValidator',
    'InferenceEngine',
    'ABTestManager',
    'BatchInferenceEngine',
    'InferenceMonitor',
    'InferenceAPI'
]