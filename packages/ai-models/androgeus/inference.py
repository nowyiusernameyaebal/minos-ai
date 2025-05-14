"""
        Run the complete inference pipeline on OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data.
            confidence_threshold: Threshold for confidence in trading signals.
            
        Returns:
            Dictionary with trading signals and insights.
            
        Raises:
            ValueError: If there are issues with input data or model.
            RuntimeError: If inference fails.
        """
        try:
            # Preprocess data
            logger.info("Preprocessing market data...")
            preprocessed_data = self.preprocess_data(ohlcv_data)
            
            # Prepare model inputs
            logger.info("Preparing model inputs...")
            model_inputs, num_samples = self.prepare_model_inputs(preprocessed_data)
            
            # Generate predictions
            logger.info("Generating predictions...")
            predictions = self.generate_predictions(model_inputs)
            
            # Postprocess predictions
            logger.info("Postprocessing predictions...")
            results = self.postprocess_predictions(
                predictions, ohlcv_data, confidence_threshold
            )
            
            logger.info("Inference completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Failed to run inference: {str(e)}")
    
    def batch_inference(self, 
                       ohlcv_data: pd.DataFrame, 
                       window_size: int = 20,
                       step_size: int = 1,
                       confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Run inference on multiple windows of OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data.
            window_size: Number of bars to include in each inference window.
            step_size: Number of bars to step between inference windows.
            confidence_threshold: Threshold for confidence in trading signals.
            
        Returns:
            DataFrame with inference results for each window.
            
        Raises:
            ValueError: If there are issues with input data or parameters.
        """
        if len(ohlcv_data) < window_size:
            logger.error(f"Not enough data for window size {window_size}")
            raise ValueError(f"Need at least {window_size} data points, got {len(ohlcv_data)}")
        
        # Calculate number of windows
        num_windows = (len(ohlcv_data) - window_size) // step_size + 1
        
        # Initialize results list
        results_list = []
        
        # Run inference for each window
        for i in range(num_windows):
            # Extract window data
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_data = ohlcv_data.iloc[start_idx:end_idx].copy()
            
            try:
                # Run inference on window
                window_result = self.run_inference(window_data, confidence_threshold)
                
                # Add window information
                window_result['window_start'] = ohlcv_data.index[start_idx]
                window_result['window_end'] = ohlcv_data.index[end_idx - 1]
                
                # Append to results list
                results_list.append(window_result)
                
            except Exception as e:
                logger.warning(f"Failed inference for window {i}: {str(e)}")
                # Continue with next window
        
        # Convert results to DataFrame
        # First flatten the nested dictionaries
        flattened_results = []
        for result in results_list:
            flat_result = {
                'timestamp': result['timestamp'],
                'window_start': result['window_start'],
                'window_end': result['window_end'],
                'last_close': result['last_close'],
                'recommendation': result['recommendation']
            }
            
            # Add signal and insight data
            if 'signals' in result:
                for signal_name, signal_data in result['signals'].items():
                    for key, value in signal_data.items():
                        flat_result[f'signal_{signal_name}_{key}'] = value
            
            if 'insights' in result:
                for insight_name, insight_data in result['insights'].items():
                    if isinstance(insight_data, dict):
                        for key, value in insight_data.items():
                            flat_result[f'insight_{insight_name}_{key}'] = value
                    else:
                        flat_result[f'insight_{insight_name}'] = insight_data
            
            flattened_results.append(flat_result)
        
        # Create DataFrame
        results_df = pd.DataFrame(flattened_results)
        
        logger.info(f"Batch inference completed with {len(results_df)} windows")
        return results_df
    
    def analyze_market_conditions(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market conditions using technical indicators.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data.
            
        Returns:
            Dictionary with market condition analysis.
            
        Raises:
            ValueError: If there are issues with input data.
        """
        # Ensure we have enough data
        min_data_points = 200  # Need enough data for long-term indicators
        if len(ohlcv_data) < min_data_points:
            logger.warning(f"Limited data for market analysis. "
                          f"Got {len(ohlcv_data)} points, recommended at least {min_data_points}")
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(ohlcv_data)
        
        # Initialize results
        analysis = {
            'timestamp': datetime.now(),
            'market_conditions': {},
            'trend_analysis': {},
            'volatility_analysis': {},
            'momentum_analysis': {},
            'support_resistance': {},
            'summary': {}
        }
        
        # Get the most recent values
        latest_data = preprocessed_data.iloc[-1]
        
        # Trend analysis
        trend_indicators = {
            'sma_20': 'Short-term trend',
            'sma_50': 'Medium-term trend',
            'sma_200': 'Long-term trend',
            'trend_strength': 'Trend strength'
        }
        
        for indicator, description in trend_indicators.items():
            if indicator in preprocessed_data.columns:
                # Determine trend direction
                close = ohlcv_data['close'].iloc[-1]
                
                if indicator.startswith('sma_'):
                    # Compare price to SMA
                    sma_value = preprocessed_data[indicator].iloc[-1]
                    
                    if close > sma_value:
                        direction = 'bullish'
                        strength = (close / sma_value - 1) * 100
                    elif close < sma_value:
                        direction = 'bearish'
                        strength = (1 - close / sma_value) * 100
                    else:
                        direction = 'neutral'
                        strength = 0
                    
                    analysis['trend_analysis'][indicator] = {
                        'description': description,
                        'direction': direction,
                        'strength': float(strength),
                        'value': float(sma_value)
                    }
                elif indicator == 'trend_strength':
                    # Use trend strength indicator
                    strength_value = preprocessed_data[indicator].iloc[-1]
                    
                    # Determine if strong trend is present
                    if strength_value > 0.7:
                        trend_status = 'strong'
                    elif strength_value > 0.3:
                        trend_status = 'moderate'
                    else:
                        trend_status = 'weak'
                    
                    analysis['trend_analysis'][indicator] = {
                        'description': description,
                        'status': trend_status,
                        'value': float(strength_value)
                    }
        
        # Volatility analysis
        volatility_indicators = {
            'atr_14': 'Average True Range',
            'bollinger_upper': 'Bollinger Upper Band',
            'bollinger_lower': 'Bollinger Lower Band',
            'volatility_ratio_14': 'Volatility Ratio'
        }
        
        for indicator, description in volatility_indicators.items():
            if indicator in preprocessed_data.columns:
                value = preprocessed_data[indicator].iloc[-1]
                
                if indicator == 'atr_14':
                    # Normalize ATR by price
                    close = ohlcv_data['close'].iloc[-1]
                    atr_percent = (value / close) * 100
                    
                    if atr_percent > 3:
                        volatility = 'high'
                    elif atr_percent > 1:
                        volatility = 'medium'
                    else:
                        volatility = 'low'
                    
                    analysis['volatility_analysis'][indicator] = {
                        'description': description,
                        'volatility': volatility,
                        'value': float(value),
                        'percent': float(atr_percent)
                    }
                elif indicator.startswith('bollinger_'):
                    analysis['volatility_analysis'][indicator] = {
                        'description': description,
                        'value': float(value)
                    }
                elif indicator == 'volatility_ratio_14':
                    if value > 1.2:
                        vol_status = 'increasing'
                    elif value < 0.8:
                        vol_status = 'decreasing'
                    else:
                        vol_status = 'stable'
                    
                    analysis['volatility_analysis'][indicator] = {
                        'description': description,
                        'status': vol_status,
                        'value': float(value)
                    }
        
        # Add Bollinger Bandwidth if bands are available
        if 'bollinger_upper' in preprocessed_data.columns and 'bollinger_lower' in preprocessed_data.columns:
            upper = preprocessed_data['bollinger_upper'].iloc[-1]
            lower = preprocessed_data['bollinger_lower'].iloc[-1]
            middle = (upper + lower) / 2
            
            bandwidth = (upper - lower) / middle
            
            if bandwidth > 0.1:
                band_status = 'wide'
            elif bandwidth < 0.03:
                band_status = 'narrow'
            else:
                band_status = 'normal'
            
            analysis['volatility_analysis']['bollinger_bandwidth'] = {
                'description': 'Bollinger Bandwidth',
                'status': band_status,
                'value': float(bandwidth)
            }
        
        # Momentum analysis
        momentum_indicators = {
            'rsi_14': 'Relative Strength Index',
            'macd': 'MACD Line',
            'macd_histogram': 'MACD Histogram',
            'price_momentum_14': 'Price Momentum'
        }
        
        for indicator, description in momentum_indicators.items():
            if indicator in preprocessed_data.columns:
                value = preprocessed_data[indicator].iloc[-1]
                
                if indicator == 'rsi_14':
                    if value > 70:
                        status = 'overbought'
                    elif value < 30:
                        status = 'oversold'
                    else:
                        status = 'neutral'
                    
                    analysis['momentum_analysis'][indicator] = {
                        'description': description,
                        'status': status,
                        'value': float(value)
                    }
                elif indicator.startswith('macd'):
                    if indicator == 'macd':
                        if value > 0:
                            status = 'bullish'
                        else:
                            status = 'bearish'
                    elif indicator == 'macd_histogram':
                        if value > 0:
                            status = 'bullish_momentum'
                        else:
                            status = 'bearish_momentum'
                    
                    analysis['momentum_analysis'][indicator] = {
                        'description': description,
                        'status': status,
                        'value': float(value)
                    }
                elif indicator == 'price_momentum_14':
                    if value > 0.02:
                        status = 'strong_bullish'
                    elif value > 0:
                        status = 'bullish'
                    elif value < -0.02:
                        status = 'strong_bearish'
                    else:
                        status = 'bearish'
                    
                    analysis['momentum_analysis'][indicator] = {
                        'description': description,
                        'status': status,
                        'value': float(value)
                    }
        
        # Calculate support and resistance levels using pivot points
        if len(ohlcv_data) >= 20:
            # Get pivot points for the latest data
            pivots = self.indicators.pivot_points(ohlcv_data)
            
            # Add pivot points to analysis
            for level, value in pivots.items():
                analysis['support_resistance'][level] = float(value.iloc[-1])
            
            # Identify key levels
            close = ohlcv_data['close'].iloc[-1]
            
            # Find closest support and resistance
            levels = sorted([(level, float(value.iloc[-1])) for level, value in pivots.items()], key=lambda x: x[1])
            supports = [(level, value) for level, value in levels if value < close]
            resistances = [(level, value) for level, value in levels if value > close]
            
            if supports:
                closest_support = max(supports, key=lambda x: x[1])
                analysis['support_resistance']['closest_support'] = {
                    'level': closest_support[0],
                    'value': closest_support[1]
                }
            
            if resistances:
                closest_resistance = min(resistances, key=lambda x: x[1])
                analysis['support_resistance']['closest_resistance'] = {
                    'level': closest_resistance[0],
                    'value': closest_resistance[1]
                }
        
        # Generate market conditions summary
        # 1. Determine overall trend
        overall_trend = 'neutral'
        if 'sma_50' in analysis['trend_analysis'] and 'sma_200' in analysis['trend_analysis']:
            sma_50 = analysis['trend_analysis']['sma_50']['direction']
            sma_200 = analysis['trend_analysis']['sma_200']['direction']
            
            if sma_50 == 'bullish' and sma_200 == 'bullish':
                overall_trend = 'bullish'
            elif sma_50 == 'bearish' and sma_200 == 'bearish':
                overall_trend = 'bearish'
            elif sma_50 == 'bullish' and sma_200 == 'bearish':
                overall_trend = 'cautiously_bullish'
            elif sma_50 == 'bearish' and sma_200 == 'bullish':
                overall_trend = 'cautiously_bearish'
        
        # 2. Determine overall momentum
        overall_momentum = 'neutral'
        if 'rsi_14' in analysis['momentum_analysis'] and 'macd' in analysis['momentum_analysis']:
            rsi = analysis['momentum_analysis']['rsi_14']['status']
            macd = analysis['momentum_analysis']['macd']['status']
            
            if rsi == 'overbought' and macd == 'bullish':
                overall_momentum = 'strongly_bullish'
            elif rsi == 'oversold' and macd == 'bearish':
                overall_momentum = 'strongly_bearish'
            elif (rsi == 'neutral' or rsi == 'overbought') and macd == 'bullish':
                overall_momentum = 'bullish'
            elif (rsi == 'neutral' or rsi == 'oversold') and macd == 'bearish':
                overall_momentum = 'bearish'
            elif rsi == 'overbought':
                overall_momentum = 'overbought'
            elif rsi == 'oversold':
                overall_momentum = 'oversold'
        
        # 3. Determine overall volatility
        overall_volatility = 'normal'
        if 'atr_14' in analysis['volatility_analysis']:
            volatility = analysis['volatility_analysis']['atr_14']['volatility']
            overall_volatility = volatility
        
        if 'bollinger_bandwidth' in analysis['volatility_analysis']:
            bandwidth = analysis['volatility_analysis']['bollinger_bandwidth']['status']
            if bandwidth == 'narrow':
                overall_volatility = f"{overall_volatility}_contracting"
            elif bandwidth == 'wide':
                overall_volatility = f"{overall_volatility}_expanding"
        
        # 4. Add market regime
        if overall_trend == 'bullish' and overall_volatility != 'high':
            market_regime = 'uptrend'
        elif overall_trend == 'bearish' and overall_volatility != 'high':
            market_regime = 'downtrend'
        elif overall_volatility == 'high':
            market_regime = 'volatile'
        elif overall_volatility.endswith('contracting'):
            market_regime = 'consolidation'
        else:
            market_regime = 'range_bound'
        
        # Add to summary
        analysis['summary'] = {
            'trend': overall_trend,
            'momentum': overall_momentum,
            'volatility': overall_volatility,
            'market_regime': market_regime
        }
        
        return analysis
    
    def generate_strategy_parameters(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading strategy parameters based on market conditions.
        
        Args:
            market_analysis: Dictionary with market condition analysis.
            
        Returns:
            Dictionary with strategy parameters.
        """
        # Extract market conditions from analysis
        trend = market_analysis['summary']['trend']
        momentum = market_analysis['summary']['momentum']
        volatility = market_analysis['summary']['volatility']
        market_regime = market_analysis['summary']['market_regime']
        
        # Initialize strategy parameters
        strategy_params = {
            'timestamp': datetime.now(),
            'risk_profile': {},
            'entry_conditions': {},
            'exit_conditions': {},
            'position_sizing': {},
            'timeframes': {},
            'indicators': {},
            'description': ''
        }
        
        # Set risk profile based on market conditions
        if market_regime == 'uptrend':
            risk_profile = 'moderate'
        elif market_regime == 'downtrend':
            risk_profile = 'conservative'
        elif market_regime == 'volatile':
            risk_profile = 'very_conservative'
        elif market_regime == 'consolidation':
            risk_profile = 'conservative'
        else:  # range_bound
            risk_profile = 'moderate'
        
        strategy_params['risk_profile'] = {
            'profile': risk_profile,
            'max_position_size': self._get_max_position_size(risk_profile),
            'stop_loss_percent': self._get_stop_loss_percent(risk_profile, volatility),
            'take_profit_percent': self._get_take_profit_percent(risk_profile, volatility)
        }
        
        # Set entry conditions based on market regime
        if market_regime == 'uptrend':
            strategy_params['entry_conditions'] = {
                'direction': 'long',
                'confirmation_indicators': ['price_above_sma_50', 'rsi_above_40', 'macd_bullish'],
                'filters': ['avoid_overbought']
            }
        elif market_regime == 'downtrend':
            strategy_params['entry_conditions'] = {
                'direction': 'short',
                'confirmation_indicators': ['price_below_sma_50', 'rsi_below_60', 'macd_bearish'],
                'filters': ['avoid_oversold']
            }
        elif market_regime == 'volatile':
            strategy_params['entry_conditions'] = {
                'direction': 'both',
                'confirmation_indicators': ['bollinger_band_bounce', 'volume_spike'],
                'filters': ['require_strong_signal']
            }
        elif market_regime == 'consolidation':
            strategy_params['entry_conditions'] = {
                'direction': 'both',
                'confirmation_indicators': ['bollinger_band_breakout', 'volume_confirmation'],
                'filters': ['wait_for_confirmation']
            }
        else:  # range_bound
            strategy_params['entry_conditions'] = {
                'direction': 'both',
                'confirmation_indicators': ['rsi_extremes', 'price_at_support_resistance'],
                'filters': ['trending_timeframe_alignment']
            }
        
        # Set exit conditions
        strategy_params['exit_conditions'] = {
            'stop_loss': True,
            'take_profit': True,
            'trailing_stop': self._should_use_trailing_stop(market_regime),
            'exit_indicators': self._get_exit_indicators(market_regime, trend)
        }
        
        # Set position sizing
        strategy_params['position_sizing'] = {
            'method': 'volatility_adjusted',
            'max_risk_per_trade': self._get_max_risk_per_trade(risk_profile),
            'base_position_size': self._get_base_position_size(risk_profile)
        }
        
        # Set timeframes to monitor
        strategy_params['timeframes'] = self._get_timeframes(market_regime)
        
        # Set indicators to use
        strategy_params['indicators'] = self._get_indicators(market_regime, trend)
        
        # Generate strategy description
        strategy_params['description'] = self._generate_strategy_description(
            market_regime, trend, strategy_params
        )
        
        return strategy_params
    
    def _get_max_position_size(self, risk_profile: str) -> float:
        """
        Get maximum position size based on risk profile.
        """
        if risk_profile == 'very_conservative':
            return 0.05  # 5% of portfolio
        elif risk_profile == 'conservative':
            return 0.1   # 10% of portfolio
        elif risk_profile == 'moderate':
            return 0.15  # 15% of portfolio
        elif risk_profile == 'aggressive':
            return 0.25  # 25% of portfolio
        else:
            return 0.1   # Default to conservative
    
    def _get_stop_loss_percent(self, risk_profile: str, volatility: str) -> float:
        """
        Get stop loss percentage based on risk profile and volatility.
        """
        base_stop = 0.0
        
        # Base stop loss by risk profile
        if risk_profile == 'very_conservative':
            base_stop = 0.02  # 2%
        elif risk_profile == 'conservative':
            base_stop = 0.03  # 3%
        elif risk_profile == 'moderate':
            base_stop = 0.05  # 5%
        elif risk_profile == 'aggressive':
            base_stop = 0.08  # 8%
        else:
            base_stop = 0.03  # Default to conservative
        
        # Adjust for volatility
        if volatility.startswith('high'):
            return base_stop * 1.5
        elif volatility.startswith('low'):
            return base_stop * 0.8
        else:
            return base_stop
    
    def _get_take_profit_percent(self, risk_profile: str, volatility: str) -> float:
        """
        Get take profit percentage based on risk profile and volatility.
        """
        # Get stop loss as reference
        stop_loss = self._get_stop_loss_percent(risk_profile, volatility)
        
        # Set risk-reward ratio based on risk profile
        if risk_profile == 'very_conservative':
            return stop_loss * 1.5  # 1.5:1 reward-to-risk
        elif risk_profile == 'conservative':
            return stop_loss * 2.0  # 2:1 reward-to-risk
        elif risk_profile == 'moderate':
            return stop_loss * 2.5  # 2.5:1 reward-to-risk
        elif risk_profile == 'aggressive':
            return stop_loss * 3.0  # 3:1 reward-to-risk
        else:
            return stop_loss * 2.0  # Default to 2:1
    
    def _should_use_trailing_stop(self, market_regime: str) -> bool:
        """
        Determine if trailing stop should be used based on market regime.
        """
        return market_regime in ['uptrend', 'downtrend', 'volatile']
    
    def _get_exit_indicators(self, market_regime: str, trend: str) -> List[str]:
        """
        Get exit indicators based on market regime and trend.
        """
        if market_regime == 'uptrend':
            return ['rsi_overbought', 'macd_bearish_cross', 'bearish_engulfing']
        elif market_regime == 'downtrend':
            return ['rsi_oversold', 'macd_bullish_cross', 'bullish_engulfing']
        elif market_regime == 'volatile':
            return ['parabolic_move', 'volume_climax', 'volatility_contraction']
        elif market_regime == 'consolidation':
            return ['breakout_failure', 'volume_dry_up']
        else:  # range_bound
            return ['price_rejection', 'overbought_oversold_extremes']
    
    def _get_max_risk_per_trade(self, risk_profile: str) -> float:
        """
        Get maximum risk per trade based on risk profile.
        """
        if risk_profile == 'very_conservative':
            return 0.005  # 0.5% of portfolio
        elif risk_profile == 'conservative':
            return 0.01   # 1% of portfolio
        elif risk_profile == 'moderate':
            return 0.02   # 2% of portfolio
        elif risk_profile == 'aggressive':
            return 0.03   # 3% of portfolio
        else:
            return 0.01   # Default to conservative
    
    def _get_base_position_size(self, risk_profile: str) -> float:
        """
        Get base position size based on risk profile.
        """
        if risk_profile == 'very_conservative':
            return 0.03  # 3% of portfolio
        elif risk_profile == 'conservative':
            return 0.05  # 5% of portfolio
        elif risk_profile == 'moderate':
            return 0.1   # 10% of portfolio
        elif risk_profile == 'aggressive':
            return 0.15  # 15% of portfolio
        else:
            return 0.05  # Default to conservative
    
    def _get_timeframes(self, market_regime: str) -> Dict[str, bool]:
        """
        Get timeframes to monitor based on market regime.
        """
        if market_regime == 'uptrend' or market_regime == 'downtrend':
            return {
                'intraday': True,
                'daily': True,
                'weekly': True,
                'monthly': False
            }
        elif market_regime == 'volatile':
            return {
                'intraday': True,
                'daily': True,
                'weekly': False,
                'monthly': False
            }
        elif market_regime == 'consolidation':
            return {
                'intraday': True,
                'daily': True,
                'weekly': False,
                'monthly': False
            }
        else:  # range_bound
            return {
                'intraday': True,
                'daily': True,
                'weekly': True,
                'monthly': False
            }
    
    def _get_indicators(self, market_regime: str, trend: str) -> Dict[str, List[str]]:
        """
        Get indicators to use based on market regime and trend.
        """
        indicators = {
            'primary': [],
            'secondary': [],
            'confirmation': []
        }
        
        if market_regime == 'uptrend':
            indicators['primary'] = ['sma_20', 'sma_50', 'macd']
            indicators['secondary'] = ['rsi', 'volume_pressure', 'supertrend']
            indicators['confirmation'] = ['engulfing_pattern', 'trend_strength']
        elif market_regime == 'downtrend':
            indicators['primary'] = ['sma_20', 'sma_50', 'macd']
            indicators['secondary'] = ['rsi', 'volume_pressure', 'supertrend']
            indicators['confirmation'] = ['engulfing_pattern', 'trend_strength']
        elif market_regime == 'volatile':
            indicators['primary'] = ['bollinger_bands', 'atr', 'volatility_ratio']
            indicators['secondary'] = ['rsi', 'stoch', 'vortex']
            indicators['confirmation'] = ['volume_spike', 'pivot_points']
        elif market_regime == 'consolidation':
            indicators['primary'] = ['bollinger_bands', 'keltner_channel', 'donchian_channel']
            indicators['secondary'] = ['rsi', 'mfi', 'obv']
            indicators['confirmation'] = ['doji_pattern', 'volume_dry_up']
        else:  # range_bound
            indicators['primary'] = ['support_resistance', 'pivot_points', 'rsi']
            indicators['secondary'] = ['stoch', 'williams', 'cci']
            indicators['confirmation'] = ['hammer_pattern', 'engulfing_pattern']
        
        return indicators
    
    def _generate_strategy_description(self, 
                                      market_regime: str, 
                                      trend: str,
                                      strategy_params: Dict[str, Any]) -> str:
        """
        Generate a human-readable strategy description.
        """
        risk_profile = strategy_params['risk_profile']['profile']
        direction = strategy_params['entry_conditions']['direction']
        
        if market_regime == 'uptrend':
            base_description = (
                f"This is a {risk_profile} trend-following strategy designed for an "
                f"uptrend market environment. It focuses on {direction} positions "
            )
        elif market_regime == 'downtrend':
            base_description = (
                f"This is a {risk_profile} trend-following strategy designed for a "
                f"downtrend market environment. It focuses on {direction} positions "
            )
        elif market_regime == 'volatile':
            base_description = (
                f"This is a {risk_profile} volatility-focused strategy designed for a "
                f"high-volatility market environment. It allows for {direction} positions "
            )
        elif market_regime == 'consolidation':
            base_description = (
                f"This is a {risk_profile} breakout strategy designed for a "
                f"consolidating market environment. It looks for {direction} opportunities "
            )
        else:  # range_bound
            base_description = (
                f"This is a {risk_profile} range-trading strategy designed for a "
                f"range-bound market environment. It trades {direction} from support and resistance "
            )
        
        # Entry and exit description
        entry_desc = "Entry signals combine "
        entry_desc += ", ".join(strategy_params['entry_conditions']['confirmation_indicators'])
        
        exit_desc = "Exits include "
        if strategy_params['exit_conditions']['stop_loss']:
            exit_desc += "stop losses, "
        if strategy_params['exit_conditions']['take_profit']:
            exit_desc += "take profits, "
        if strategy_params['exit_conditions']['trailing_stop']:
            exit_desc += "trailing stops, "
        
        exit_desc += "and indicator-based exits using "
        exit_desc += ", ".join(strategy_params['exit_conditions']['exit_indicators'])
        
        # Position sizing description
        position_desc = (
            f"Position sizing uses a {strategy_params['position_sizing']['method']} approach "
            f"with maximum risk of {strategy_params['position_sizing']['max_risk_per_trade'] * 100}% "
            f"per trade and base position size of {strategy_params['position_sizing']['base_position_size'] * 100}% "
            f"of portfolio."
        )
        
        # Combine descriptions
        full_description = (
            f"{base_description}. {entry_desc}. {exit_desc}. {position_desc} "
            f"Recommended timeframes include {', '.join([tf for tf, use in strategy_params['timeframes'].items() if use])}."
        )
        
        return full_description
        
    def export_report(self, 
                     market_analysis: Dict[str, Any], 
                     strategy_params: Dict[str, Any],
                     prediction_results: Dict[str, Any] = None,
                     file_format: str = 'json',
                     file_path: Optional[str] = None) -> str:
        """
        Export a comprehensive report with market analysis and strategy recommendations.
        
        Args:
            market_analysis: Dictionary with market condition analysis.
            strategy_params: Dictionary with strategy parameters.
            prediction_results: Optional dictionary with prediction results.
            file_format: Report format ('json', 'csv', or 'html').
            file_path: Path to save the report file. If None, a default path will be used.
            
        Returns:
            Path to the exported report file.
            
        Raises:
            ValueError: If an invalid file format is specified.
        """
        # Create report dictionary
        report = {
            'timestamp': datetime.now().isoformat(),
            'market_analysis': market_analysis,
            'strategy_parameters': strategy_params
        }
        
        if prediction_results:
            report['prediction_results'] = prediction_results
        
        # Generate default file path if not provided
        if file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"androgeus_report_{timestamp}.{file_format}"
            file_path = os.path.join(current_dir, 'reports', filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export report based on requested format
        if file_format == 'json':
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif file_format == 'csv':
            # Flatten the report structure
            flattened_data = self._flatten_dict(report)
            pd.DataFrame([flattened_data]).to_csv(file_path, index=False)
        elif file_format == 'html':
            # Create HTML report
            html_content = self._generate_html_report(report)
            with open(file_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Invalid file format: {file_format}. "
                           f"Supported formats are 'json', 'csv', and 'html'")
        
        logger.info(f"Report exported to {file_path}")
        return file_path
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten.
            parent_key: Parent key for nested dictionaries.
            sep: Separator between keys.
            
        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}{sep}{i}", sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        Generate an HTML report.
        
        Args:
            report: Report dictionary.
            
        Returns:
            HTML report content.
        """
        # Generate simple HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Androgeus Market Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333366; }
                .section { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .subsection { margin-left: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .bullish { color: green; }
                .bearish { color: red; }
                .neutral { color: gray; }
                .warning { color: orange; }
                .timestamp { font-style: italic; color: #666; }
            </style>
        </head>
        <body>
            <h1>Androgeus Market Analysis Report</h1>
            <p class="timestamp">Generated on: %s</p>
        """ % report['timestamp']
        
        # Add market analysis section
        if 'market_analysis' in report:
            market = report['market_analysis']
            html_content += """
            <div class="section">
                <h2>Market Analysis</h2>
            """
            
            # Add summary
            if 'summary' in market:
                html_content += """
                <div class="subsection">
                    <h3>Market Summary</h3>
                    <table>
                        <tr><th>Factor</th><th>Status</th></tr>
                """
                
                for factor, status in market['summary'].items():
                    css_class = ""
                    if "bullish" in str(status).lower():
                        css_class = "bullish"
                    elif "bearish" in str(status).lower():
                        css_class = "bearish"
                    elif "neutral" in str(status).lower():
                        css_class = "neutral"
                    
                    html_content += f"""
                        <tr><td>{factor.capitalize()}</td><td class="{css_class}">{status}</td></tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add trend analysis
            if 'trend_analysis' in market:
                html_content += """
                <div class="subsection">
                    <h3>Trend Analysis</h3>
                    <table>
                        <tr><th>Indicator</th><th>Description</th><th>Status</th><th>Value</th></tr>
                """
                
                for indicator, data in market['trend_analysis'].items():
                    if isinstance(data, dict):
                        description = data.get('description', indicator)
                        
                        # Determine status and CSS class
                        if 'direction' in data:
                            status = data['direction']
                            css_class = "bullish" if status == "bullish" else "bearish" if status == "bearish" else "neutral"
                        elif 'status' in data:
                            status = data['status']
                            css_class = ""
                        else:
                            status = "N/A"
                            css_class = ""
                        
                        # Get value
                        value = data.get('value', "N/A")
                        
                        html_content += f"""
                            <tr>
                                <td>{indicator}</td>
                                <td>{description}</td>
                                <td class="{css_class}">{status}</td>
                                <td>{value:.4f if isinstance(value, float) else value}</td>
                            </tr>
                        """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add volatility analysis
            if 'volatility_analysis' in market:
                html_content += """
                <div class="subsection">
                    <h3>Volatility Analysis</h3>
                    <table>
                        <tr><th>Indicator</th><th>Description</th><th>Status</th><th>Value</th></tr>
                """
                
                for indicator, data in market['volatility_analysis'].items():
                    if isinstance(data, dict):
                        description = data.get('description', indicator)
                        
                        # Determine status and CSS class
                        if 'volatility' in data:
                            status = data['volatility']
                            css_class = "warning" if status == "high" else "neutral" if status == "medium" else ""
                        elif 'status' in data:
                            status = data['status']
                            css_class = "warning" if "increasing" in status else ""
                        else:
                            status = "N/A"
                            css_class = ""
                        
                        # Get value
                        value = data.get('value', "N/A")
                        
                        html_content += f"""
                            <tr>
                                <td>{indicator}</td>
                                <td>{description}</td>
                                <td class="{css_class}">{status}</td>
                                <td>{value:.4f if isinstance(value, float) else value}</td>
                            </tr>
                        """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add momentum analysis
            if 'momentum_analysis' in market:
                html_content += """
                <div class="subsection">
                    <h3>Momentum Analysis</h3>
                    <table>
                        <tr><th>Indicator</th><th>Description</th><th>Status</th><th>Value</th></tr>
                """
                
                for indicator, data in market['momentum_analysis'].items():
                    if isinstance(data, dict):
                        description = data.get('description', indicator)
                        
                        # Determine status and CSS class
                        if 'status' in data:
                            status = data['status']
                            if "bullish" in status:
                                css_class = "bullish"
                            elif "bearish" in status:
                                css_class = "bearish"
                            elif status == "overbought":
                                css_class = "warning"
                            elif status == "oversold":
                                css_class = "warning"
                            else:
                                css_class = "neutral"
                        else:
                            status = "N/A"
                            css_class = ""
                        
                        # Get value
                        value = data.get('value', "N/A")
                        
                        html_content += f"""
                            <tr>
                                <td>{indicator}</td>
                                <td>{description}</td>
                                <td class="{css_class}">{status}</td>
                                <td>{value:.4f if isinstance(value, float) else value}</td>
                            </tr>
                        """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add support/resistance
            if 'support_resistance' in market:
                html_content += """
                <div class="subsection">
                    <h3>Support & Resistance Levels</h3>
                    <table>
                        <tr><th>Level</th><th>Value</th></tr>
                """
                
                for level, value in market['support_resistance'].items():
                    if isinstance(value, dict):
                        level_name = f"{level} ({value.get('level', '')})"
                        level_value = value.get('value', "N/A")
                    else:
                        level_name = level
                        level_value = value
                    
                    html_content += f"""
                        <tr>
                            <td>{level_name}</td>
                            <td>{level_value:.4f if isinstance(level_value, float) else level_value}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        # Add strategy parameters section
        if 'strategy_parameters' in report:
            strategy = report['strategy_parameters']
            html_content += """
            <div class="section">
                <h2>Trading Strategy Parameters</h2>
                <p>%s</p>
            """ % strategy.get('description', '')
            
            # Add risk profile
            if 'risk_profile' in strategy:
                html_content += """
                <div class="subsection">
                    <h3>Risk Profile</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                """
                
                for param, value in strategy['risk_profile'].items():
                    html_content += f"""
                        <tr>
                            <td>{param.replace('_', ' ').capitalize()}</td>
                            <td>{value:.2%f if isinstance(value, float) and 'percent' in param else value}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add entry conditions
            if 'entry_conditions' in strategy:
                html_content += """
                <div class="subsection">
                    <h3>Entry Conditions</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                """
                
                for param, value in strategy['entry_conditions'].items():
                    if isinstance(value, list):
                        value_str = ", ".join(value)
                    else:
                        value_str = str(value)
                    
                    html_content += f"""
                        <tr>
                            <td>{param.replace('_', ' ').capitalize()}</td>
                            <td>{value_str}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add exit conditions
            if 'exit_conditions' in strategy:
                html_content += """
                <div class="subsection">
                    <h3>Exit Conditions</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                """
                
                for param, value in strategy['exit_conditions'].items():
                    if isinstance(value, list):
                        value_str = ", ".join(value)
                    else:
                        value_str = str(value)
                    
                    html_content += f"""
                        <tr>
                            <td>{param.replace('_', ' ').capitalize()}</td>
                            <td>{value_str}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add position sizing
            if 'position_sizing' in strategy:
                html_content += """
                <div class="subsection">
                    <h3>Position Sizing</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                """
                
                for param, value in strategy['position_sizing'].items():
                    html_content += f"""
                        <tr>
                            <td>{param.replace('_', ' ').capitalize()}</td>
                            <td>{value:.2%f if isinstance(value, float) and 'risk' in param else value}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        # Add prediction results section if available
        if 'prediction_results' in report:
            prediction = report['prediction_results']
            html_content += """
            <div class="section">
                <h2>Price Predictions</h2>
            """
            
            # Add signals
            if 'signals' in prediction:
                html_content += """
                <div class="subsection">
                    <h3>Trading Signals</h3>
                    <table>
                        <tr><th>Signal</th><th>Direction</th><th>Confidence</th></tr>
                """
                
                for signal_name, signal_data in prediction['signals'].items():
                    if isinstance(signal_data, dict):
                        direction = signal_data.get('signal', 'N/A')
                        confidence = signal_data.get('confidence', 'N/A')
                        
                        # Determine CSS class
                        css_class = ""
                        if direction == 'buy':
                            css_class = "bullish"
                        elif direction == 'sell':
                            css_class = "bearish"
                        
                        html_content += f"""
                            <tr>
                                <td>{signal_name}</td>
                                <td class="{css_class}">{direction}</td>
                                <td>{confidence:.2%f if isinstance(confidence, float) else confidence}</td>
                            </tr>
                        """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add insights
            if 'insights' in prediction:
                html_content += """
                <div class="subsection">
                    <h3>Market Insights</h3>
                    <table>
                        <tr><th>Insight</th><th>Value</th></tr>
                """
                
                for insight_name, insight_data in prediction['insights'].items():
                    if isinstance(insight_data, dict):
                        for key, value in insight_data.items():
                            html_content += f"""
                                <tr>
                                    <td>{insight_name.replace('_', ' ').capitalize()} - {key}</td>
                                    <td>{value:.4f if isinstance(value, float) else value}</td>
                                </tr>
                            """
                    else:
                        html_content += f"""
                            <tr>
                                <td>{insight_name.replace('_', ' ').capitalize()}</td>
                                <td>{insight_data:.4f if isinstance(insight_data, float) else insight_data}</td>
                            </tr>
                        """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add recommendation
            if 'recommendation' in prediction:
                recommendation = prediction['recommendation']
                
                # Determine CSS class
                css_class = ""
                if "buy" in recommendation:
                    css_class = "bullish"
                elif "sell" in recommendation:
                    css_class = "bearish"
                
                html_content += f"""
                <div class="subsection">
                    <h3>Final Recommendation</h3>
                    <p class="{css_class}"><strong>{recommendation.upper()}</strong></p>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        # Close HTML document
        html_content += """
        </body>
        </html>
        """
        
        return html_content


# Example usage when running module directly
if __name__ == "__main__":
    # Load example data
    try:
        from ..utils.data_loader import load_sample_data
        
        # Load sample data
        ohlcv_data = load_sample_data()
        
        # Initialize inference engine
        inference = AndrogeusInference()
        
        # Run inference
        results = inference.run_inference(ohlcv_data)
        
        # Print results
        print("\nTrading recommendation:", results['recommendation'])
        print("Direction signal:", results['signals']['direction']['signal'])
        print("Confidence:", results['signals']['direction']['confidence'])
        
        if 'target_price' in results['insights']:
            print("Target price:", results['insights']['target_price'])
        
        # Run market analysis
        market_analysis = inference.analyze_market_conditions(ohlcv_data)
        
        # Generate strategy parameters
        strategy_params = inference.generate_strategy_parameters(market_analysis)
        
        # Export report
        report_path = inference.export_report(market_analysis, strategy_params, results)
        print(f"\nReport exported to: {report_path}")
        
    except ImportError:
        print("Sample data loader not available. Using random data instead.")
        
        # Generate random data
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate random OHLCV data
        np.random.seed(42)
        n_samples = 200
        
        # Create date index
        date_index = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
        
        # Generate random walk price
        close = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        
        # Generate OHLC based on close
        high = close + np.random.uniform(0, 2, n_samples)
        low = close - np.random.uniform(0, 2, n_samples)
        open_price = low + np.random.uniform(0, high - low, n_samples)
        
        # Generate volume
        volume = np.random.uniform(1000, 5000, n_samples)
        
        # Create DataFrame
        ohlcv_data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=date_index)
        
        # Initialize inference engine
        inference = AndrogeusInference()
        
        # Run inference
        results = inference.run_inference(ohlcv_data)
        
        # Print results
        print("\nTrading recommendation:", results['recommendation'])
        print("Direction signal:", results['signals']['direction']['signal'])
        print("Confidence:", results['signals']['direction']['confidence'])
        
        if 'target_price' in results['insights']:
            print("Target price:", results['insights']['target_price'])
            
        # Run market analysis
        market_analysis = inference.analyze_market_conditions(ohlcv_data)
        
        # Generate strategy parameters
        strategy_params = inference.generate_strategy_parameters(market_analysis)
        
        # Export report
        report_path = inference.export_report(market_analysis, strategy_params, results)
        print(f"\nReport exported to: {report_path}")"""
Androgeus Inference Module

This module provides functionality for performing inference with the Androgeus
technical analysis AI model. It handles data preprocessing, model execution,
and post-processing of results to generate trading signals and recommendations.

The inference process includes:
1. Loading and preprocessing market data
2. Calculating technical indicators
3. Feature normalization and selection
4. Running the trained model to generate predictions
5. Post-processing results and generating trading signals

Author: Minos-AI Team
Date: January 8, 2025
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# Import Androgeus modules
from .model import AndrogeusModel
from .indicators import TechnicalIndicators
from ..utils.data_loader import load_ohlcv_data, standardize_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AndrogeusInference:
    """
    Inference engine for the Androgeus technical analysis AI model.
    
    This class handles loading the trained model, processing market data,
    generating predictions, and providing trading signals based on the model's output.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 indicators_config_path: Optional[str] = None):
        """
        Initialize the Androgeus inference engine.
        
        Args:
            model_path: Path to the trained model file. If None, a default path will be used.
            config_path: Path to the model configuration file. If None, a default path will be used.
            indicators_config_path: Path to the indicators configuration file. If None,
                                   a default set of indicators will be used.
                                   
        Raises:
            FileNotFoundError: If the model or configuration files cannot be found.
            ValueError: If there are issues loading the model or configurations.
        """
        self.model = None
        self.indicators = TechnicalIndicators()
        self.config = None
        self.indicators_config = None
        
        # Load configuration files
        self._load_configurations(config_path, indicators_config_path)
        
        # Load the trained model
        self._load_model(model_path)
        
        logger.info("Androgeus inference engine initialized successfully")
    
    def _load_configurations(self, config_path: Optional[str], indicators_config_path: Optional[str]) -> None:
        """
        Load configuration files for the model and indicators.
        
        Args:
            config_path: Path to the model configuration file.
            indicators_config_path: Path to the indicators configuration file.
            
        Raises:
            FileNotFoundError: If configuration files cannot be found.
            json.JSONDecodeError: If configuration files are not valid JSON.
        """
        # Load model configuration
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config.json')
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Model configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.error(f"Model configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in model configuration file at {config_path}")
            raise
        
        # Load indicators configuration if provided
        if indicators_config_path is not None:
            try:
                with open(indicators_config_path, 'r') as f:
                    self.indicators_config = json.load(f)
                logger.info(f"Indicators configuration loaded from {indicators_config_path}")
            except FileNotFoundError:
                logger.error(f"Indicators configuration file not found at {indicators_config_path}")
                raise
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in indicators configuration file at {indicators_config_path}")
                raise
        else:
            # Use default indicators configuration
            self.indicators_config = {
                "selected_indicators": [
                    "sma", "ema", "rsi", "macd", "bollinger", "atr",
                    "stoch", "adx", "obv", "cmf", "ichimoku", "price_momentum",
                    "trend_strength", "volatility_ratio"
                ],
                "periods": {
                    "sma": [20, 50, 200],
                    "ema": [20, 50, 200],
                    "rsi": [14],
                    "macd": [12],
                    "bollinger": [20],
                    "atr": [14],
                    "stoch": [14],
                    "adx": [14],
                    "obv": [1],
                    "cmf": [20],
                    "ichimoku": [9],
                    "price_momentum": [14],
                    "trend_strength": [14],
                    "volatility_ratio": [14]
                },
                "normalization_method": "z-score"
            }
            logger.info("Using default indicators configuration")
    
    def _load_model(self, model_path: Optional[str]) -> None:
        """
        Load the trained Androgeus model.
        
        Args:
            model_path: Path to the trained model file.
            
        Raises:
            FileNotFoundError: If the model file cannot be found.
            ValueError: If there are issues loading the model.
        """
        try:
            if model_path is None:
                # Try to find model in standard location
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_models = [
                    os.path.join(current_dir, 'saved_models', 'androgeus_model.h5'),
                    os.path.join(current_dir, 'saved_models', 'androgeus_model'),
                    os.path.join(current_dir, '..', '..', 'models', 'androgeus', 'androgeus_model.h5'),
                    os.path.join(current_dir, '..', '..', 'models', 'androgeus', 'androgeus_model')
                ]
                
                for path in possible_models:
                    if os.path.exists(path):
                        model_path = path
                        break
            
            if model_path is None or not os.path.exists(model_path):
                logger.warning("Model file not found. Running in feature extraction mode only.")
                # Create a fresh model instance without loading weights
                self.model = AndrogeusModel(config_path=None)
                return
            
            # Load the trained model
            self.model = AndrogeusModel.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def preprocess_data(self, 
                        ohlcv_data: pd.DataFrame,
                        calculate_indicators: bool = True) -> pd.DataFrame:
        """
        Preprocess market data for model inference.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data.
            calculate_indicators: Whether to calculate technical indicators.
            
        Returns:
            Preprocessed DataFrame with calculated indicators.
            
        Raises:
            ValueError: If OHLCV data is missing required columns.
        """
        # Ensure OHLCV data has the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in ohlcv_data.columns:
                logger.error(f"OHLCV data is missing required column: {col}")
                raise ValueError(f"OHLCV data must contain column: {col}")
        
        # Standardize column names and order
        ohlcv_data = standardize_dataframe(ohlcv_data)
        
        # Calculate indicators if requested
        if calculate_indicators:
            indicators_data = self.indicators.calculate_all(
                ohlcv_data,
                selected_indicators=self.indicators_config.get("selected_indicators"),
                periods=self.indicators_config.get("periods")
            )
            
            # Normalize indicators
            # Skip the OHLCV columns (first 5 columns)
            indicators_only = indicators_data.iloc[:, 5:]
            normalized_indicators = self.indicators.normalize_features(
                indicators_only,
                method=self.indicators_config.get("normalization_method", "z-score")
            )
            
            # Combine OHLCV with normalized indicators
            preprocessed_data = pd.concat([ohlcv_data, normalized_indicators], axis=1)
        else:
            preprocessed_data = ohlcv_data
        
        return preprocessed_data
    
    def prepare_model_inputs(self, preprocessed_data: pd.DataFrame) -> Tuple[List[np.ndarray], int]:
        """
        Prepare input arrays for the model.
        
        Args:
            preprocessed_data: DataFrame with preprocessed data.
            
        Returns:
            Tuple containing:
            - List of numpy arrays for model inputs
            - Number of valid samples
            
        Raises:
            ValueError: If there's not enough data for model inputs.
        """
        # Get configuration parameters
        sequence_length = self.config.get('sequence_length', 50)
        price_features = self.config.get('price_features', 5)  # OHLCV
        indicator_features = self.config.get('indicator_features', 20)
        use_metadata = self.config.get('use_metadata', False)
        metadata_features = self.config.get('metadata_features', 10) if use_metadata else 0
        
        # Check if we have enough data
        if len(preprocessed_data) < sequence_length:
            logger.error(f"Not enough data for sequence length {sequence_length}")
            raise ValueError(f"Need at least {sequence_length} data points, got {len(preprocessed_data)}")
        
        # Prepare price sequence input
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        price_data = preprocessed_data[price_columns].values
        
        # Determine the number of valid samples
        num_samples = len(preprocessed_data) - sequence_length + 1
        
        # Create price sequences
        price_sequences = np.zeros((num_samples, sequence_length, price_features))
        for i in range(num_samples):
            price_sequences[i] = price_data[i:i+sequence_length, :price_features]
        
        # Prepare technical indicator input
        # Get all columns except OHLCV
        indicator_columns = preprocessed_data.columns.difference(price_columns)
        
        # If we have more indicators than the model expects, select the most important ones
        if len(indicator_columns) > indicator_features:
            logger.warning(f"Too many indicators ({len(indicator_columns)}), "
                          f"selecting {indicator_features} most important ones")
            
            # Prioritize predefined set of important indicators
            priority_indicators = [
                'rsi', 'macd', 'bollinger', 'sma', 'ema', 'adx', 'atr',
                'volatility_ratio', 'trend_strength', 'price_momentum'
            ]
            
            # Add columns that contain these priority indicators first
            selected_indicators = []
            for priority in priority_indicators:
                matching_cols = [col for col in indicator_columns if priority in col]
                selected_indicators.extend(matching_cols)
                if len(selected_indicators) >= indicator_features:
                    break
            
            # If we still need more, add the remaining indicators
            if len(selected_indicators) < indicator_features:
                remaining = list(set(indicator_columns) - set(selected_indicators))
                selected_indicators.extend(remaining[:indicator_features - len(selected_indicators)])
            
            # Trim to the required number
            indicator_columns = selected_indicators[:indicator_features]
        
        # If we have fewer indicators than the model expects, pad with zeros
        if len(indicator_columns) < indicator_features:
            logger.warning(f"Too few indicators ({len(indicator_columns)}), "
                          f"padding to {indicator_features}")
            
            # Padding will be done implicitly by initializing indicator_data with zeros
            
        # Extract indicator data for the most recent time step
        indicator_data = np.zeros((num_samples, indicator_features))
        for i, col in enumerate(indicator_columns[:indicator_features]):
            # Use the most recent value for each sequence
            indicator_data[:, i] = preprocessed_data[col].values[sequence_length-1:]
        
        # Prepare metadata input if required
        metadata_data = None
        if use_metadata:
            # Metadata features could include market-wide data, sector data, etc.
            # In this implementation, we use dummy values for illustration
            metadata_data = np.zeros((num_samples, metadata_features))
        
        # Prepare model inputs
        model_inputs = [price_sequences, indicator_data]
        if metadata_data is not None:
            model_inputs.append(metadata_data)
        
        return model_inputs, num_samples
    
    def generate_predictions(self, model_inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Generate predictions using the model.
        
        Args:
            model_inputs: List of numpy arrays for model inputs.
            
        Returns:
            Dictionary mapping output names to prediction arrays.
            
        Raises:
            ValueError: If the model has not been loaded.
            RuntimeError: If prediction fails.
        """
        if self.model is None:
            logger.error("Cannot generate predictions: Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Get raw predictions from model
            raw_predictions = self.model.predict(model_inputs)
            
            # Format predictions into a dictionary
            predictions = {}
            
            # Map predictions to output names
            output_idx = 0
            
            # Check which outputs are enabled
            if self.config.get('predict_direction', True):
                predictions['direction'] = raw_predictions[output_idx]
                output_idx += 1
            
            if self.config.get('predict_magnitude', True):
                predictions['magnitude'] = raw_predictions[output_idx]
                output_idx += 1
            
            if self.config.get('predict_price_targets', False):
                predictions['price_targets'] = raw_predictions[output_idx]
                output_idx += 1
            
            if self.config.get('predict_volatility', False):
                predictions['volatility'] = raw_predictions[output_idx]
                output_idx += 1
            
            logger.info(f"Generated predictions for {len(predictions)} outputs")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise RuntimeError(f"Failed to generate predictions: {str(e)}")
    
    def postprocess_predictions(self, 
                               predictions: Dict[str, np.ndarray],
                               ohlcv_data: pd.DataFrame,
                               confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Postprocess model predictions to generate trading signals and insights.
        
        Args:
            predictions: Dictionary of raw model predictions.
            ohlcv_data: Original OHLCV data.
            confidence_threshold: Threshold for confidence in trading signals.
            
        Returns:
            Dictionary with trading signals and insights.
        """
        # Get the most recent close price
        last_close = ohlcv_data['close'].iloc[-1]
        
        # Initialize results dictionary
        results = {
            'timestamp': datetime.now(),
            'last_close': last_close,
            'signals': {},
            'insights': {}
        }
        
        # Process direction predictions
        if 'direction' in predictions:
            direction_prob = predictions['direction'][-1][0]
            
            # Convert probability to signal
            if direction_prob > confidence_threshold:
                signal = 'buy'
                confidence = direction_prob
            elif direction_prob < (1 - confidence_threshold):
                signal = 'sell'
                confidence = 1 - direction_prob
            else:
                signal = 'neutral'
                confidence = 0.5 - abs(0.5 - direction_prob)
            
            results['signals']['direction'] = {
                'signal': signal,
                'confidence': float(confidence),
                'raw_probability': float(direction_prob)
            }
            
            # Add insight about expected direction
            expected_direction = "up" if direction_prob > 0.5 else "down"
            results['insights']['expected_direction'] = {
                'direction': expected_direction,
                'probability': float(max(direction_prob, 1 - direction_prob))
            }
        
        # Process magnitude predictions
        if 'magnitude' in predictions:
            predicted_magnitude = predictions['magnitude'][-1][0]
            
            # Add magnitude to results
            results['insights']['expected_change'] = {
                'percent': float(predicted_magnitude * 100),
                'price_change': float(last_close * predicted_magnitude)
            }
            
            # If both direction and magnitude available, calculate target price
            if 'direction' in predictions:
                direction_prob = predictions['direction'][-1][0]
                expected_direction = 1 if direction_prob > 0.5 else -1
                target_price = last_close * (1 + expected_direction * abs(predicted_magnitude))
                
                results['insights']['target_price'] = float(target_price)
        
        # Process price targets predictions
        if 'price_targets' in predictions:
            price_targets = predictions['price_targets'][-1]
            
            # Calculate actual price targets
            target_prices = []
            for i, target_pct in enumerate(price_targets):
                target_price = last_close * (1 + target_pct)
                target_prices.append(float(target_price))
            
            results['insights']['price_targets'] = target_prices
        
        # Process volatility predictions
        if 'volatility' in predictions:
            predicted_volatility = predictions['volatility'][-1][0]
            
            # Add volatility to results
            results['insights']['expected_volatility'] = {
                'percent': float(predicted_volatility * 100)
            }
            
            # If expected_change is available, calculate range
            if 'expected_change' in results['insights']:
                expected_change = results['insights']['expected_change']['percent'] / 100
                volatility_range = {
                    'upper': float(last_close * (1 + expected_change + predicted_volatility)),
                    'lower': float(last_close * (1 + expected_change - predicted_volatility))
                }
                
                results['insights']['price_range'] = volatility_range
        
        # Generate final trading recommendation
        if 'direction' in predictions:
            direction_signal = results['signals']['direction']['signal']
            direction_confidence = results['signals']['direction']['confidence']
            
            if direction_signal == 'buy' and direction_confidence > 0.7:
                recommendation = 'strong_buy'
            elif direction_signal == 'buy':
                recommendation = 'buy'
            elif direction_signal == 'sell' and direction_confidence > 0.7:
                recommendation = 'strong_sell'
            elif direction_signal == 'sell':
                recommendation = 'sell'
            else:
                recommendation = 'hold'
            
            results['recommendation'] = recommendation
        else:
            results['recommendation'] = 'unavailable'
        
        return results
    
    def run_inference(self, 
                     ohlcv_data: pd.DataFrame, 
                     confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Run the complete inference pipeline on OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data.
            confidence_threshold: Threshold for confidence in trading signals.
            
        Returns:
            Dictionary with trading signals and insights