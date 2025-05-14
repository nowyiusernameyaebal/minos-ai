def volume_weighted_average_price(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            df: OHLCV DataFrame.
            period: VWAP period.
            
        Returns:
            Series with VWAP values.
        """
        # Calculate typical price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price * volume
        pv = tp * df['volume']
        
        # Calculate cumulative price * volume and cumulative volume
        cumulative_pv = pv.rolling(window=period).sum()
        cumulative_volume = df['volume'].rolling(window=period).sum()
        
        # Calculate VWAP
        vwap = cumulative_pv / cumulative_volume
        
        return vwap
    
    def price_volume_trend(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calculate Price Volume Trend (PVT).
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with PVT values.
        """
        close = df['close']
        volume = df['volume']
        
        # Calculate price percentage change
        price_change_pct = close.pct_change()
        
        # Calculate PVT values
        pvt_values = price_change_pct * volume
        
        # Cumulative sum to get PVT
        pvt = pvt_values.cumsum()
        
        return pvt
    
    def chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Args:
            df: OHLCV DataFrame.
            period: CMF period.
            
        Returns:
            Series with CMF values.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        
        # Replace NaN values (when high = low)
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate CMF
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    def accumulation_distribution(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line (A/D Line).
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with A/D Line values.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        
        # Replace NaN values (when high = low)
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate A/D Line
        ad_line = mfv.cumsum()
        
        return ad_line
    
    def ease_of_movement(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Ease of Movement (EOM).
        
        Args:
            df: OHLCV DataFrame.
            period: EOM smoothing period.
            
        Returns:
            Series with EOM values.
        """
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Calculate distance moved
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # Calculate box ratio
        box_ratio = (volume / 1000000) / (high - low)
        
        # Calculate raw EOM
        raw_eom = distance / box_ratio
        
        # Smooth EOM
        eom = raw_eom.rolling(window=period).mean()
        
        return eom
    
    #############################
    # PRICE ACTION INDICATORS
    #############################
    
    def heikin_ashi(self, df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candles.
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            DataFrame with Heikin-Ashi OHLC values.
        """
        ha = pd.DataFrame(index=df.index)
        
        # Calculate Heikin-Ashi Close
        ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Initialize first Heikin-Ashi Open with first Open
        ha['open'] = df['open'].iloc[0]
        
        # Calculate remaining Heikin-Ashi Open values
        for i in range(1, len(df)):
            ha['open'].iloc[i] = (ha['open'].iloc[i-1] + ha['close'].iloc[i-1]) / 2
        
        # Calculate Heikin-Ashi High and Low
        ha['high'] = pd.concat([df['high'], ha['open'], ha['close']], axis=1).max(axis=1)
        ha['low'] = pd.concat([df['low'], ha['open'], ha['close']], axis=1).min(axis=1)
        
        return ha
    
    def parabolic_sar(self, df: pd.DataFrame, period: float = 0.02) -> pd.Series:
        """
        Calculate Parabolic SAR.
        
        Args:
            df: OHLCV DataFrame.
            period: Initial acceleration factor (default 0.02).
            
        Returns:
            Series with Parabolic SAR values.
        """
        high = df['high']
        low = df['low']
        
        # Initialize parameters
        af = period  # Acceleration factor
        af_step = period  # Acceleration factor step
        af_max = 0.2  # Maximum acceleration factor
        
        # Initialize SAR with first low for uptrend, first high for downtrend
        # For simplicity, we'll assume we start in an uptrend
        is_uptrend = True
        sar = pd.Series(index=df.index)
        sar.iloc[0] = low.iloc[0]
        
        # Initialize extreme points
        ep = high.iloc[0]  # Extreme point
        
        # Calculate SAR values
        for i in range(1, len(df)):
            # Update SAR
            if is_uptrend:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                # Ensure SAR is not higher than the lows of the last two periods
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                
                # Check if trend reverses
                if low.iloc[i] < sar.iloc[i]:
                    is_uptrend = False
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_step
                else:
                    # Update extreme point and acceleration factor
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_step, af_max)
            else:
                sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
                
                # Ensure SAR is not lower than the highs of the last two periods
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                
                # Check if trend reverses
                if high.iloc[i] > sar.iloc[i]:
                    is_uptrend = True
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_step
                else:
                    # Update extreme point and acceleration factor
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_step, af_max)
        
        return sar
    
    def ichimoku_cloud(self, df: pd.DataFrame, period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            df: OHLCV DataFrame.
            period: Tenkan-sen period (fast). Other periods will be scaled accordingly.
            
        Returns:
            Dictionary with Ichimoku Cloud components.
        """
        # Define periods
        tenkan_period = period
        kijun_period = period * 3  # Usually 26
        senkou_span_b_period = period * 5  # Usually 52
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = df['high'].rolling(window=senkou_span_b_period).max()
        senkou_low = df['low'].rolling(window=senkou_span_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def supertrend(self, df: pd.DataFrame, period: int = 10) -> Dict[str, pd.Series]:
        """
        Calculate SuperTrend indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: SuperTrend period.
            
        Returns:
            Dictionary with SuperTrend values and direction.
        """
        # Default multiplier
        multiplier = 3.0
        
        # Calculate ATR
        atr = self.average_true_range(df, period)
        
        # Calculate basic upper and lower bands
        hl2 = (df['high'] + df['low']) / 2
        basic_upper_band = hl2 + (multiplier * atr)
        basic_lower_band = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend with first value
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate first SuperTrend value
        supertrend.iloc[period] = (
            basic_lower_band.iloc[period] 
            if df['close'].iloc[period] > basic_upper_band.iloc[period]
            else basic_upper_band.iloc[period]
        )
        
        # Calculate SuperTrend values
        for i in range(period + 1, len(df)):
            # Calculate current upper and lower bands
            if basic_upper_band.iloc[i] < supertrend.iloc[i-1] or df['close'].iloc[i-1] > supertrend.iloc[i-1]:
                upper_band = basic_upper_band.iloc[i]
            else:
                upper_band = supertrend.iloc[i-1]
            
            if basic_lower_band.iloc[i] > supertrend.iloc[i-1] or df['close'].iloc[i-1] < supertrend.iloc[i-1]:
                lower_band = basic_lower_band.iloc[i]
            else:
                lower_band = supertrend.iloc[i-1]
            
            # Determine trend direction
            if supertrend.iloc[i-1] == upper_band:
                # Previous trend was down
                if df['close'].iloc[i] > upper_band:
                    # Trend reversal to up
                    supertrend.iloc[i] = lower_band
                    direction.iloc[i] = 1
                else:
                    # Continue downtrend
                    supertrend.iloc[i] = upper_band
                    direction.iloc[i] = -1
            else:
                # Previous trend was up
                if df['close'].iloc[i] < lower_band:
                    # Trend reversal to down
                    supertrend.iloc[i] = upper_band
                    direction.iloc[i] = -1
                else:
                    # Continue uptrend
                    supertrend.iloc[i] = lower_band
                    direction.iloc[i] = 1
        
        return {
            'supertrend': supertrend,
            'direction': direction
        }
    
    #############################
    # PATTERN RECOGNITION
    #############################
    
    def engulfing_pattern(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Detect bullish and bearish engulfing patterns.
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with pattern signals (1 for bullish, -1 for bearish, 0 for none).
        """
        # Initialize result series
        engulfing = pd.Series(0, index=df.index)
        
        # Calculate body sizes
        body_size = abs(df['close'] - df['open'])
        
        # Check for bullish engulfing
        bullish_engulfing = (
            (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['open'] <= df['close'].shift(1)) &  # Current open is lower than previous close
            (df['close'] >= df['open'].shift(1)) &  # Current close is higher than previous open
            (body_size > body_size.shift(1))  # Current body is larger than previous body
        )
        
        # Check for bearish engulfing
        bearish_engulfing = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['open'] > df['close']) &  # Current candle is bearish
            (df['open'] >= df['close'].shift(1)) &  # Current open is higher than previous close
            (df['close'] <= df['open'].shift(1)) &  # Current close is lower than previous open
            (body_size > body_size.shift(1))  # Current body is larger than previous body
        )
        
        # Assign signals
        engulfing[bullish_engulfing] = 1
        engulfing[bearish_engulfing] = -1
        
        return engulfing
    
    def hammer_pattern(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Detect hammer and shooting star patterns.
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with pattern signals (1 for hammer, -1 for shooting star, 0 for none).
        """
        # Initialize result series
        hammer = pd.Series(0, index=df.index)
        
        # Calculate body size and shadows
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        # Check for hammer pattern
        is_hammer = (
            (lower_shadow > 2 * body_size) &  # Long lower shadow
            (upper_shadow < 0.2 * body_size) &  # Short upper shadow
            (body_size > 0)  # Non-zero body size
        )
        
        # Check for shooting star pattern
        is_shooting_star = (
            (upper_shadow > 2 * body_size) &  # Long upper shadow
            (lower_shadow < 0.2 * body_size) &  # Short lower shadow
            (body_size > 0)  # Non-zero body size
        )
        
        # Assign signals
        hammer[is_hammer] = 1
        hammer[is_shooting_star] = -1
        
        return hammer
    
    def doji_pattern(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Detect doji patterns.
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with doji signals (1 for doji, 0 for none).
        """
        # Initialize result series
        doji = pd.Series(0, index=df.index)
        
        # Calculate body size and total candle size
        body_size = abs(df['close'] - df['open'])
        candle_size = df['high'] - df['low']
        
        # Check for doji pattern
        is_doji = (
            (body_size < 0.1 * candle_size) &  # Very small body compared to total candle
            (candle_size > 0)  # Non-zero candle size
        )
        
        # Assign signals
        doji[is_doji] = 1
        
        return doji
    
    def pivot_points(self, df: pd.DataFrame, period: int = 1) -> Dict[str, pd.Series]:
        """
        Calculate pivot points and support/resistance levels.
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Dictionary with pivot points and support/resistance levels.
        """
        # Initialize result dictionary
        pivot_levels = {}
        
        # Get previous day's data
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        # Calculate pivot point (PP)
        pp = (prev_high + prev_low + prev_close) / 3
        pivot_levels['pp'] = pp
        
        # Calculate support levels
        s1 = (2 * pp) - prev_high
        s2 = pp - (prev_high - prev_low)
        s3 = s1 - (prev_high - prev_low)
        
        pivot_levels['s1'] = s1
        pivot_levels['s2'] = s2
        pivot_levels['s3'] = s3
        
        # Calculate resistance levels
        r1 = (2 * pp) - prev_low
        r2 = pp + (prev_high - prev_low)
        r3 = r1 + (prev_high - prev_low)
        
        pivot_levels['r1'] = r1
        pivot_levels['r2'] = r2
        pivot_levels['r3'] = r3
        
        return pivot_levels
    
    #############################
    # STATISTICAL INDICATORS
    #############################
    
    def z_score(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Z-score for price.
        
        Args:
            df: OHLCV DataFrame.
            period: Z-score period.
            
        Returns:
            Series with Z-score values.
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        # Calculate Z-score
        z_score = (df['close'] - rolling_mean) / rolling_std
        
        return z_score
    
    def rolling_variance(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate rolling variance of price.
        
        Args:
            df: OHLCV DataFrame.
            period: Variance period.
            
        Returns:
            Series with variance values.
        """
        # Calculate rolling variance
        variance = df['close'].rolling(window=period).var()
        
        return variance
    
    def rolling_kurtosis(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate rolling kurtosis of price returns.
        
        Args:
            df: OHLCV DataFrame.
            period: Kurtosis period.
            
        Returns:
            Series with kurtosis values.
        """
        # Calculate price returns
        returns = df['close'].pct_change()
        
        # Calculate rolling kurtosis
        kurtosis = returns.rolling(window=period).apply(
            lambda x: stats.kurtosis(x, fisher=True), raw=True
        )
        
        return kurtosis
    
    def rolling_skewness(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate rolling skewness of price returns.
        
        Args:
            df: OHLCV DataFrame.
            period: Skewness period.
            
        Returns:
            Series with skewness values.
        """
        # Calculate price returns
        returns = df['close'].pct_change()
        
        # Calculate rolling skewness
        skewness = returns.rolling(window=period).apply(
            lambda x: stats.skew(x), raw=True
        )
        
        return skewness
    
    #############################
    # MARKET MICROSTRUCTURE INDICATORS
    #############################
    
    def volume_synchronized_probability_of_informed_trading(self, df: pd.DataFrame, period: int = 50) -> pd.Series:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
        
        Args:
            df: OHLCV DataFrame.
            period: VPIN bucket period.
            
        Returns:
            Series with VPIN values.
        """
        # Calculate price changes
        price_change = df['close'].diff()
        
        # Classify volume as buy or sell based on price change
        buy_volume = np.where(price_change > 0, df['volume'], 0)
        sell_volume = np.where(price_change < 0, df['volume'], 0)
        
        # When price doesn't change, split volume equally
        unchanged_volume = np.where(price_change == 0, df['volume'] / 2, 0)
        buy_volume += unchanged_volume
        sell_volume += unchanged_volume
        
        # Calculate order imbalance
        order_imbalance = abs(buy_volume - sell_volume)
        
        # Calculate rolling sum of volume and imbalance
        total_volume = pd.Series(df['volume']).rolling(window=period).sum()
        total_imbalance = pd.Series(order_imbalance).rolling(window=period).sum()
        
        # Calculate VPIN
        vpin = total_imbalance / total_volume
        
        return vpin
    
    def kyle_lambda(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Kyle's Lambda (price impact coefficient).
        
        Args:
            df: OHLCV DataFrame.
            period: Rolling regression period.
            
        Returns:
            Series with Kyle's Lambda values.
        """
        # Calculate price changes and signed volume
        price_change = df['close'].diff()
        signed_volume = df['volume'] * np.sign(price_change)
        
        # Initialize result series
        lambda_values = pd.Series(np.nan, index=df.index)
        
        # Calculate Lambda for each window
        for i in range(period, len(df)):
            x = signed_volume.iloc[i-period+1:i+1].values.reshape(-1, 1)
            y = price_change.iloc[i-period+1:i+1].values
            
            # Add constant for regression
            x = np.hstack([np.ones((x.shape[0], 1)), x])
            
            try:
                # Perform linear regression
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                
                # Lambda is the coefficient on signed volume
                lambda_values.iloc[i] = beta[1]
            except:
                # If regression fails, keep as NaN
                pass
        
        return lambda_values
    
    def amihud_illiquidity(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Amihud Illiquidity Ratio.
        
        Args:
            df: OHLCV DataFrame.
            period: Illiquidity period.
            
        Returns:
            Series with Amihud Illiquidity values.
        """
        # Calculate absolute returns
        returns = df['close'].pct_change().abs()
        
        # Calculate daily dollar volume (in millions)
        dollar_volume = (df['close'] * df['volume']) / 1_000_000
        
        # Calculate return-to-volume ratio
        rtv_ratio = returns / dollar_volume
        
        # Calculate rolling average of ratio
        amihud = rtv_ratio.rolling(window=period).mean()
        
        return amihud
    
    #############################
    # CROSS-ASSET INDICATORS
    #############################
    
    def rolling_correlation(self, df: pd.DataFrame, reference_series: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between price and reference series.
        
        Args:
            df: OHLCV DataFrame.
            reference_series: Series to calculate correlation against.
            period: Correlation period.
            
        Returns:
            Series with correlation values.
        """
        # Ensure the reference series has the same index
        if not df.index.equals(reference_series.index):
            raise ValueError("Reference series must have the same index as the OHLCV DataFrame")
        
        # Calculate rolling correlation
        correlation = df['close'].rolling(window=period).corr(reference_series)
        
        return correlation
    
    def relative_performance(self, df: pd.DataFrame, reference_series: pd.Series, period: int = 1) -> pd.Series:
        """
        Calculate relative performance compared to reference series.
        
        Args:
            df: OHLCV DataFrame.
            reference_series: Series to calculate relative performance against.
            period: Not used, included for API consistency.
            
        Returns:
            Series with relative performance values.
        """
        # Ensure the reference series has the same index
        if not df.index.equals(reference_series.index):
            raise ValueError("Reference series must have the same index as the OHLCV DataFrame")
        
        # Normalize both series to start at 1
        normalized_price = df['close'] / df['close'].iloc[0]
        normalized_reference = reference_series / reference_series.iloc[0]
        
        # Calculate relative performance
        relative_performance = normalized_price / normalized_reference
        
        return relative_performance
    
    def rolling_beta(self, df: pd.DataFrame, reference_series: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling beta with respect to reference series.
        
        Args:
            df: OHLCV DataFrame.
            reference_series: Series to calculate beta against (e.g., market returns).
            period: Beta calculation period.
            
        Returns:
            Series with beta values.
        """
        # Ensure the reference series has the same index
        if not df.index.equals(reference_series.index):
            raise ValueError("Reference series must have the same index as the OHLCV DataFrame")
        
        # Calculate returns
        price_returns = df['close'].pct_change()
        reference_returns = reference_series.pct_change()
        
        # Initialize beta series
        beta = pd.Series(np.nan, index=df.index)
        
        # Calculate beta for each window
        for i in range(period, len(df)):
            y = price_returns.iloc[i-period+1:i+1].values
            x = reference_returns.iloc[i-period+1:i+1].values
            
            # Remove NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask].reshape(-1, 1)
            y = y[mask]
            
            if len(x) > 0:
                # Add constant for regression
                x_with_const = np.hstack([np.ones((x.shape[0], 1)), x])
                
                try:
                    # Perform linear regression
                    beta_value = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
                    beta.iloc[i] = beta_value
                except:
                    # If regression fails, keep as NaN
                    pass
        
        return beta
    
    #############################
    # CUSTOM ANDROGEUS INDICATORS
    #############################
    
    def price_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Price Momentum indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Momentum period.
            
        Returns:
            Series with momentum values.
        """
        # Calculate momentum (price change over period)
        momentum = df['close'] - df['close'].shift(period)
        
        # Normalize by average price
        avg_price = df['close'].rolling(window=period).mean()
        normalized_momentum = momentum / avg_price
        
        return normalized_momentum
    
    def volatility_ratio(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Volatility Ratio indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Volatility period.
            
        Returns:
            Series with volatility ratio values.
        """
        # Calculate recent volatility (short-term)
        short_period = max(5, period // 3)
        short_volatility = df['close'].pct_change().rolling(window=short_period).std()
        
        # Calculate long-term volatility
        long_volatility = df['close'].pct_change().rolling(window=period).std()
        
        # Calculate ratio
        ratio = short_volatility / long_volatility
        
        return ratio
    
    def trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Trend Strength indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Trend period.
            
        Returns:
            Series with trend strength values.
        """
        # Calculate linear regression slope
        close = df['close'].values
        time = np.arange(len(close))
        
        # Initialize slope series
        slope = pd.Series(np.nan, index=df.index)
        
        # Calculate slope for each window
        for i in range(period, len(df)):
            y = close[i-period+1:i+1]
            x = time[i-period+1:i+1].reshape(-1, 1)
            
            # Add constant for regression
            x_with_const = np.hstack([np.ones((x.shape[0], 1)), x])
            
            try:
                # Perform linear regression
                beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
                
                # Normalize slope by average price
                avg_price = np.mean(y)
                norm_slope = beta * period / avg_price
                
                slope.iloc[i] = norm_slope
            except:
                # If regression fails, keep as NaN
                pass
        
        # Calculate R-squared to measure trend consistency
        r_squared = pd.Series(np.nan, index=df.index)
        
        for i in range(period, len(df)):
            y = close[i-period+1:i+1]
            x = time[i-period+1:i+1]
            
            try:
                correlation = np.corrcoef(x, y)[0, 1]
                r_squared.iloc[i] = correlation ** 2
            except:
                # If calculation fails, keep as NaN
                pass
        
        # Combine slope and R-squared to get trend strength
        trend_strength = slope.abs() * r_squared
        
        return trend_strength
    
    def oscillator_divergence(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Oscillator Divergence indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Divergence detection period.
            
        Returns:
            Series with divergence signals.
        """
        # Calculate RSI
        rsi = self.relative_strength_index(df, period)
        
        # Initialize result series
        divergence = pd.Series(0, index=df.index)
        
        # Find local extrema
        # For peaks: current price higher than n-periods before and after
        # For troughs: current price lower than n-periods before and after
        comparison_window = max(3, period // 5)
        
        for i in range(comparison_window, len(df) - comparison_window):
            # Check for price peaks
            if (df['close'].iloc[i] > df['close'].iloc[i-comparison_window:i].max() and
                df['close'].iloc[i] > df['close'].iloc[i+1:i+comparison_window+1].max()):
                # Price peak
                
                # Find corresponding RSI at peak
                rsi_at_peak = rsi.iloc[i]
                
                # Look for higher previous peak within last 'period' bars
                found_prev_peak = False
                for j in range(i - period, i - comparison_window):
                    if j < 0:
                        continue
                    
                    if (df['close'].iloc[j] > df['close'].iloc[j-comparison_window:j].max() and
                        df['close'].iloc[j] > df['close'].iloc[j+1:j+comparison_window+1].max()):
                        # Found previous peak
                        found_prev_peak = True
                        
                        # Check for bearish divergence
                        if df['close'].iloc[i] > df['close'].iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                            # Bearish divergence (price higher but RSI lower)
                            divergence.iloc[i] = -1
                            break
                
            # Check for price troughs
            elif (df['close'].iloc[i] < df['close'].iloc[i-comparison_window:i].min() and
                  df['close'].iloc[i] < df['close'].iloc[i+1:i+comparison_window+1].min()):
                # Price trough
                
                # Find corresponding RSI at trough
                rsi_at_trough = rsi.iloc[i]
                
                # Look for lower previous trough within last 'period' bars
                found_prev_trough = False
                for j in range(i - period, i - comparison_window):
                    if j < 0:
                        continue
                    
                    if (df['close'].iloc[j] < df['close'].iloc[j-comparison_window:j].min() and
                        df['close'].iloc[j] < df['close'].iloc[j+1:j+comparison_window+1].min()):
                        # Found previous trough
                        found_prev_trough = True
                        
                        # Check for bullish divergence
                        if df['close'].iloc[i] < df['close'].iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                            # Bullish divergence (price lower but RSI higher)
                            divergence.iloc[i] = 1
                            break
        
        return divergence
    
    def volume_pressure(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Volume Pressure indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Volume pressure period.
            
        Returns:
            Series with volume pressure values.
        """
        # Calculate price change
        price_change = df['close'].pct_change()
        
        # Calculate volume change
        volume_change = df['volume'].pct_change()
        
        # Calculate volume-weighted price change
        vw_price_change = price_change * df['volume']
        
        # Calculate sum of volume-weighted price changes over period
        cumulative_vw_price_change = vw_price_change.rolling(window=period).sum()
        
        # Calculate total volume over period
        total_volume = df['volume'].rolling(window=period).sum()
        
        # Calculate volume pressure
        volume_pressure = cumulative_vw_price_change / total_volume
        
        # Adjust for volume trends
        avg_volume = df['volume'].rolling(window=period).mean()
        volume_trend = df['volume'] / avg_volume
        
        # Combine into final indicator
        adjusted_volume_pressure = volume_pressure * volume_trend
        
        return adjusted_volume_pressure


# Example usage when running module directly
if __name__ == "__main__":
    # Create sample OHLCV data
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
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
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_index)
    
    # Initialize indicators
    indicators = TechnicalIndicators()
    
    # Calculate some indicators
    print("Calculating indicators...")
    
    # Calculate RSI
    rsi = indicators.relative_strength_index(df)
    print(f"RSI last value: {rsi.iloc[-1]:.2f}")
    
    # Calculate MACD
    macd = indicators.moving_average_convergence_divergence(df)
    print(f"MACD last value: {macd['macd'].iloc[-1]:.2f}")
    print(f"MACD Signal last value: {macd['signal'].iloc[-1]:.2f}")
    
    # Calculate Bollinger Bands
    bollinger = indicators.bollinger_bands(df)
    print(f"Bollinger Upper Band last value: {bollinger['upper'].iloc[-1]:.2f}")
    print(f"Bollinger Middle Band last value: {bollinger['middle'].iloc[-1]:.2f}")
    print(f"Bollinger Lower Band last value: {bollinger['lower'].iloc[-1]:.2f}")
    
    # Calculate multiple indicators at once
    selected_indicators = ['sma', 'rsi', 'bollinger', 'macd', 'atr']
    periods = {
        'sma': [10, 20, 50],
        'rsi': [14],
        'bollinger': [20],
        'macd': [12],
        'atr': [14]
    }
    
    all_indicators = indicators.calculate_all(df, selected_indicators, periods)
    print(f"\nCalculated {len(all_indicators.columns) - 5} indicators")  # Subtract 5 for OHLCV columns
    
    # Get feature names
    feature_names = indicators.get_feature_names(selected_indicators, periods)
    print(f"\nFeature names: {feature_names}")
    
    # Normalize indicators
    normalized = indicators.normalize_features(all_indicators.iloc[:, 5:])  # Skip OHLCV columns
    print(f"\nNormalized indicators shape: {normalized.shape}")
    
    print("\nSample done!")"""
Androgeus Technical Indicators Module

This module provides a comprehensive set of technical indicators and feature
engineering functions for the Androgeus AI agent. These indicators are used
to analyze financial markets and identify potential trading opportunities.

The indicators implemented here range from simple moving averages to complex
oscillators and volatility measures, all optimized for use with the Androgeus model.

Author: Minos-AI Team
Date: January 8, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Callable, Any
from scipy import stats, signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    A comprehensive collection of technical indicators for market analysis.
    
    This class provides methods to calculate various technical indicators
    commonly used in financial market analysis, with optimized implementations
    for speed and accuracy.
    """
    
    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        # Dictionary to store all available indicators
        self.available_indicators = self._get_available_indicators()
        logger.info(f"Initialized TechnicalIndicators with {len(self.available_indicators)} indicators")
    
    def _get_available_indicators(self) -> Dict[str, Callable]:
        """
        Get a dictionary of all available technical indicators.
        
        Returns:
            Dictionary mapping indicator names to their calculation functions.
        """
        return {
            # Trend indicators
            'sma': self.simple_moving_average,
            'ema': self.exponential_moving_average,
            'wma': self.weighted_moving_average,
            'dema': self.double_exponential_moving_average,
            'tema': self.triple_exponential_moving_average,
            'trima': self.triangular_moving_average,
            'kama': self.kaufman_adaptive_moving_average,
            'macd': self.moving_average_convergence_divergence,
            'ppo': self.percentage_price_oscillator,
            'adx': self.average_directional_index,
            
            # Momentum indicators
            'rsi': self.relative_strength_index,
            'stoch': self.stochastic_oscillator,
            'stochrsi': self.stochastic_rsi,
            'williams': self.williams_percent_r,
            'cci': self.commodity_channel_index,
            'mfi': self.money_flow_index,
            'tsi': self.true_strength_index,
            'ultimate': self.ultimate_oscillator,
            
            # Volatility indicators
            'atr': self.average_true_range,
            'bollinger': self.bollinger_bands,
            'keltner': self.keltner_channel,
            'donchian': self.donchian_channel,
            'vortex': self.vortex_indicator,
            
            # Volume indicators
            'obv': self.on_balance_volume,
            'vwap': self.volume_weighted_average_price,
            'pvt': self.price_volume_trend,
            'cmf': self.chaikin_money_flow,
            'ad': self.accumulation_distribution,
            'eom': self.ease_of_movement,
            
            # Price action indicators
            'heikin_ashi': self.heikin_ashi,
            'parabolic_sar': self.parabolic_sar,
            'ichimoku': self.ichimoku_cloud,
            'supertrend': self.supertrend,
            
            # Pattern recognition
            'engulfing': self.engulfing_pattern,
            'hammer': self.hammer_pattern,
            'doji': self.doji_pattern,
            'pivot_points': self.pivot_points,
            
            # Statistical indicators
            'zscore': self.z_score,
            'variance': self.rolling_variance,
            'kurtosis': self.rolling_kurtosis,
            'skewness': self.rolling_skewness,
            
            # Market microstructure indicators
            'vpin': self.volume_synchronized_probability_of_informed_trading,
            'kyle_lambda': self.kyle_lambda,
            'amihud': self.amihud_illiquidity,
            
            # Cross-asset indicators
            'correlation': self.rolling_correlation,
            'relative_performance': self.relative_performance,
            'beta': self.rolling_beta,
            
            # Custom Androgeus indicators
            'price_momentum': self.price_momentum,
            'volatility_ratio': self.volatility_ratio,
            'trend_strength': self.trend_strength,
            'oscillator_divergence': self.oscillator_divergence,
            'volume_pressure': self.volume_pressure
        }
    
    def calculate_all(self, 
                      ohlcv: pd.DataFrame, 
                      selected_indicators: Optional[List[str]] = None, 
                      periods: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Calculate all or selected technical indicators.
        
        Args:
            ohlcv: DataFrame with OHLCV data. Must have columns 'open', 'high', 'low', 'close', 'volume'.
            selected_indicators: List of indicator names to calculate. If None, calculate all available indicators.
            periods: Dictionary mapping indicator names to lists of periods to use.
            
        Returns:
            DataFrame with calculated indicators.
            
        Raises:
            ValueError: If OHLCV data is missing required columns.
        """
        # Verify OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in ohlcv.columns:
                error_msg = f"OHLCV data is missing required column: {col}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Create copy of DataFrame to avoid modifying the original
        df = ohlcv.copy()
        
        # If no selected indicators, use all available
        if selected_indicators is None:
            selected_indicators = list(self.available_indicators.keys())
        
        # Default periods if not provided
        if periods is None:
            periods = {}
        
        # Calculate each selected indicator
        for indicator in selected_indicators:
            try:
                if indicator in self.available_indicators:
                    # Get indicator function
                    indicator_func = self.available_indicators[indicator]
                    
                    # Get periods for this indicator
                    indicator_periods = periods.get(indicator, self._get_default_periods(indicator))
                    
                    # Calculate indicator for each period
                    for period in indicator_periods:
                        # Get indicator with suffix for period
                        if len(indicator_periods) > 1:
                            indicator_name = f"{indicator}_{period}"
                        else:
                            indicator_name = indicator
                        
                        # Call indicator function
                        result = indicator_func(df, period)
                        
                        # If result is a DataFrame, add all columns
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                if col not in df.columns:  # Avoid duplicating columns
                                    df[col] = result[col]
                        # If result is a Series or an array, add as a single column
                        elif isinstance(result, (pd.Series, np.ndarray)):
                            df[indicator_name] = result
                        # If result is a dictionary, add each item as a column
                        elif isinstance(result, dict):
                            for key, value in result.items():
                                col_name = f"{indicator_name}_{key}" if len(result) > 1 else indicator_name
                                df[col_name] = value
                
                else:
                    logger.warning(f"Indicator '{indicator}' not found and will be skipped")
            
            except Exception as e:
                logger.error(f"Error calculating indicator '{indicator}': {str(e)}")
                # Continue with next indicator
        
        # Drop NaN values that may have been introduced
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _get_default_periods(self, indicator: str) -> List[int]:
        """
        Get default periods for a given indicator.
        
        Args:
            indicator: Name of the indicator.
            
        Returns:
            List of default periods for the indicator.
        """
        # Define default periods for each indicator group
        trend_indicators = ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama']
        momentum_indicators = ['rsi', 'stoch', 'stochrsi', 'williams', 'cci', 'mfi', 'tsi']
        volatility_indicators = ['atr', 'bollinger', 'keltner', 'donchian', 'vortex']
        volume_indicators = ['obv', 'vwap', 'pvt', 'cmf', 'ad', 'eom']
        statistical_indicators = ['zscore', 'variance', 'kurtosis', 'skewness']
        
        # Return appropriate default periods based on indicator type
        if indicator in trend_indicators:
            return [14, 50, 200]
        elif indicator in momentum_indicators:
            return [14]
        elif indicator in volatility_indicators:
            return [14]
        elif indicator in volume_indicators:
            return [20]
        elif indicator in statistical_indicators:
            return [20]
        elif indicator == 'macd':
            return [12]  # MACD uses multiple periods internally
        elif indicator == 'supertrend':
            return [10]
        elif indicator == 'ichimoku':
            return [9]  # Ichimoku uses multiple periods internally
        elif indicator == 'parabolic_sar':
            return [0.02]  # Uses acceleration factor instead of period
        else:
            # Default fallback
            return [14]
    
    def get_feature_names(self, 
                        selected_indicators: Optional[List[str]] = None, 
                        periods: Optional[Dict[str, List[int]]] = None) -> List[str]:
        """
        Get the names of all features that would be generated by calculate_all.
        
        Args:
            selected_indicators: List of indicator names. If None, use all available indicators.
            periods: Dictionary mapping indicator names to lists of periods to use.
            
        Returns:
            List of feature names.
        """
        feature_names = []
        
        # If no selected indicators, use all available
        if selected_indicators is None:
            selected_indicators = list(self.available_indicators.keys())
        
        # Default periods if not provided
        if periods is None:
            periods = {}
        
        # Get feature names for each selected indicator
        for indicator in selected_indicators:
            if indicator in self.available_indicators:
                # Get periods for this indicator
                indicator_periods = periods.get(indicator, self._get_default_periods(indicator))
                
                # Get feature names for each period
                for period in indicator_periods:
                    # Add base indicator name with period suffix if multiple periods
                    if len(indicator_periods) > 1:
                        feature_names.append(f"{indicator}_{period}")
                    else:
                        feature_names.append(indicator)
                    
                    # Add additional feature names for multi-output indicators
                    if indicator in ['bollinger', 'keltner', 'donchian', 'ichimoku', 'macd']:
                        # These indicators generate multiple outputs
                        if indicator == 'bollinger':
                            if len(indicator_periods) > 1:
                                feature_names.extend([f"bollinger_{period}_upper", f"bollinger_{period}_lower"])
                            else:
                                feature_names.extend(["bollinger_upper", "bollinger_lower"])
                        elif indicator == 'keltner':
                            if len(indicator_periods) > 1:
                                feature_names.extend([f"keltner_{period}_upper", f"keltner_{period}_lower"])
                            else:
                                feature_names.extend(["keltner_upper", "keltner_lower"])
                        elif indicator == 'donchian':
                            if len(indicator_periods) > 1:
                                feature_names.extend([f"donchian_{period}_upper", f"donchian_{period}_lower"])
                            else:
                                feature_names.extend(["donchian_upper", "donchian_lower"])
                        elif indicator == 'ichimoku':
                            feature_names.extend(["ichimoku_tenkan", "ichimoku_kijun", 
                                                "ichimoku_senkou_a", "ichimoku_senkou_b", 
                                                "ichimoku_chikou"])
                        elif indicator == 'macd':
                            feature_names.extend(["macd_signal", "macd_histogram"])
        
        return feature_names
    
    def normalize_features(self, 
                         features: pd.DataFrame, 
                         method: str = 'z-score') -> pd.DataFrame:
        """
        Normalize technical indicator features.
        
        Args:
            features: DataFrame of calculated technical indicators.
            method: Normalization method to use. Options are:
                   'z-score': Standardize features to have zero mean and unit variance.
                   'min-max': Scale features to range [0, 1].
                   'robust': Scale features using median and IQR.
                   'decimal': Scale by powers of 10 to ensure values are around [-1, 1].
            
        Returns:
            DataFrame with normalized features.
            
        Raises:
            ValueError: If an invalid normalization method is specified.
        """
        if method not in ['z-score', 'min-max', 'robust', 'decimal']:
            raise ValueError(f"Invalid normalization method: {method}. "
                           f"Supported methods are 'z-score', 'min-max', 'robust', and 'decimal'.")
        
        # Create copy to avoid modifying original
        normalized = features.copy()
        
        # Apply normalization to each column
        for col in normalized.columns:
            values = normalized[col].values
            
            # Skip columns with all same values
            if np.max(values) == np.min(values):
                continue
            
            if method == 'z-score':
                # Standardize to zero mean and unit variance
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    normalized[col] = (values - mean) / std
            
            elif method == 'min-max':
                # Scale to range [0, 1]
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    normalized[col] = (values - min_val) / (max_val - min_val)
            
            elif method == 'robust':
                # Use median and IQR for robustness to outliers
                median = np.median(values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                if iqr > 0:
                    normalized[col] = (values - median) / iqr
            
            elif method == 'decimal':
                # Scale by powers of 10 to get values around [-1, 1]
                max_abs = np.max(np.abs(values))
                if max_abs > 0:
                    power = np.floor(np.log10(max_abs))
                    normalized[col] = values / (10 ** power)
        
        return normalized
    
    def feature_selection(self, 
                        features: pd.DataFrame, 
                        target: pd.Series,
                        n_features: int = 20,
                        method: str = 'mutual_info') -> List[str]:
        """
        Select most informative features for prediction.
        
        Args:
            features: DataFrame of technical indicators.
            target: Target variable (e.g., price direction or returns).
            n_features: Number of features to select.
            method: Feature selection method. Options are:
                   'mutual_info': Mutual information for regression or classification.
                   'f_test': ANOVA F-test for classification.
                   'chi2': Chi-squared test for classification.
                   'correlation': Correlation with target for regression.
                   
        Returns:
            List of selected feature names.
            
        Raises:
            ValueError: If an invalid feature selection method is specified.
            ImportError: If scikit-learn is not installed.
        """
        try:
            from sklearn.feature_selection import (
                SelectKBest, mutual_info_regression, mutual_info_classif,
                f_classif, chi2
            )
        except ImportError:
            error_msg = "scikit-learn is required for feature selection"
            logger.error(error_msg)
            raise ImportError(error_msg + ". Install it using 'pip install scikit-learn'")
        
        # Check if we have enough features
        if features.shape[1] <= n_features:
            logger.warning(f"Number of features ({features.shape[1]}) is less than or equal to "
                          f"requested number of features ({n_features}). "
                          f"Returning all features.")
            return list(features.columns)
        
        # Determine if target is continuous or categorical
        is_continuous = len(np.unique(target)) > 10
        
        # Select appropriate scoring function
        if method == 'mutual_info':
            score_func = mutual_info_regression if is_continuous else mutual_info_classif
        elif method == 'f_test':
            if is_continuous:
                logger.warning("F-test is for classification. Using mutual information for regression.")
                score_func = mutual_info_regression
            else:
                score_func = f_classif
        elif method == 'chi2':
            if is_continuous:
                logger.warning("Chi-squared test is for classification. Using mutual information for regression.")
                score_func = mutual_info_regression
            else:
                # Chi2 requires non-negative values
                min_values = features.min()
                if (min_values < 0).any():
                    logger.warning("Chi-squared test requires non-negative values. Shifting features.")
                    features = features - min_values + 1e-8
                score_func = chi2
        elif method == 'correlation':
            # For correlation method, use absolute Pearson correlation
            correlations = features.corrwith(target).abs()
            selected_features = correlations.nlargest(n_features).index.tolist()
            return selected_features
        else:
            raise ValueError(f"Invalid feature selection method: {method}. "
                           f"Supported methods are 'mutual_info', 'f_test', 'chi2', and 'correlation'.")
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=n_features)
        selector.fit(features, target)
        
        # Get selected feature names
        mask = selector.get_support()
        selected_features = features.columns[mask].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features
    
    #############################
    # TREND INDICATORS
    #############################
    
    def simple_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with SMA values.
        """
        return df['close'].rolling(window=period).mean()
    
    def exponential_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with EMA values.
        """
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def weighted_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Weighted Moving Average (WMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with WMA values.
        """
        weights = np.arange(1, period + 1)
        wma = df['close'].rolling(period).apply(
            lambda x: np.sum(weights * x) / np.sum(weights), raw=True
        )
        return wma
    
    def double_exponential_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Double Exponential Moving Average (DEMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with DEMA values.
        """
        ema = df['close'].ewm(span=period, adjust=False).mean()
        ema_of_ema = ema.ewm(span=period, adjust=False).mean()
        dema = 2 * ema - ema_of_ema
        return dema
    
    def triple_exponential_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average (TEMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with TEMA values.
        """
        ema = df['close'].ewm(span=period, adjust=False).mean()
        ema_of_ema = ema.ewm(span=period, adjust=False).mean()
        ema_of_ema_of_ema = ema_of_ema.ewm(span=period, adjust=False).mean()
        tema = 3 * ema - 3 * ema_of_ema + ema_of_ema_of_ema
        return tema
    
    def triangular_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Triangular Moving Average (TRIMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with TRIMA values.
        """
        # TRIMA is the SMA of the SMA
        sma = df['close'].rolling(window=period).mean()
        trima = sma.rolling(window=period).mean()
        return trima
    
    def kaufman_adaptive_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).
        
        Args:
            df: OHLCV DataFrame.
            period: Moving average period.
            
        Returns:
            Series with KAMA values.
        """
        close = df['close'].values
        kama = np.zeros_like(close)
        kama[:period] = np.nan
        
        # Set initial KAMA value
        kama[period-1] = close[period-1]
        
        # Fast and slow EMA constants
        fast = 2.0 / (2.0 + 1.0)
        slow = 2.0 / (30.0 + 1.0)
        
        # Loop to calculate KAMA values
        for i in range(period, len(close)):
            # Calculate efficiency ratio
            change = abs(close[i] - close[i-period])
            volatility = np.sum(abs(close[i-j] - close[i-j-1]) for j in range(period))
            
            # Avoid division by zero
            if volatility > 0:
                er = change / volatility
            else:
                er = 0
            
            # Calculate smoothing constant
            sc = (er * (fast - slow) + slow) ** 2
            
            # Calculate KAMA
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        
        return pd.Series(kama, index=df.index)
    
    def moving_average_convergence_divergence(self, df: pd.DataFrame, period: int = 12) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            df: OHLCV DataFrame.
            period: Fast EMA period. Slow period will be 26 and signal period 9.
            
        Returns:
            Dictionary with MACD line, signal line, and histogram.
        """
        # Default parameters
        fast_period = period
        slow_period = 26
        signal_period = 9
        
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def percentage_price_oscillator(self, df: pd.DataFrame, period: int = 12) -> pd.Series:
        """
        Calculate Percentage Price Oscillator (PPO).
        
        Args:
            df: OHLCV DataFrame.
            period: Fast EMA period. Slow period will be 26.
            
        Returns:
            Series with PPO values.
        """
        # Default parameters
        fast_period = period
        slow_period = 26
        
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate PPO (similar to MACD but in percentage terms)
        ppo = 100 * (fast_ema - slow_ema) / slow_ema
        
        return ppo
    
    def average_directional_index(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: OHLCV DataFrame.
            period: ADX period.
            
        Returns:
            Dictionary with ADX, +DI, and -DI values.
        """
        # Create copies to avoid modifying original
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range
        tr = np.zeros_like(close)
        tr[1:] = np.maximum(
            np.maximum(
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1])
            ),
            np.abs(low[1:] - close[:-1])
        )
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        # +DM and -DM
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        
        plus_dm[1:] = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        )
        
        minus_dm[1:] = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        )
        
        # Convert to pandas Series with proper indexing
        tr_series = pd.Series(tr, index=df.index)
        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)
        
        # Smooth TR, +DM, and -DM with Wilder's smoothing
        smoothed_tr = tr_series.rolling(window=period).sum()
        smoothed_plus_dm = plus_dm_series.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm_series.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    #############################
    # MOMENTUM INDICATORS
    #############################
    
    def relative_strength_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: OHLCV DataFrame.
            period: RSI period.
            
        Returns:
            Series with RSI values.
        """
        delta = df['close'].diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic_oscillator(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: OHLCV DataFrame.
            period: Stochastic period.
            
        Returns:
            Dictionary with %K and %D values.
        """
        # Calculate %K
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (3-day SMA of %K)
        d = k.rolling(window=3).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    def stochastic_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Stochastic RSI.
        
        Args:
            df: OHLCV DataFrame.
            period: RSI period.
            
        Returns:
            Series with Stochastic RSI values.
        """
        # Calculate RSI
        rsi = self.relative_strength_index(df, period)
        
        # Calculate Stochastic RSI
        lowest_rsi = rsi.rolling(window=period).min()
        highest_rsi = rsi.rolling(window=period).max()
        
        stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
        
        return stoch_rsi
    
    def williams_percent_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            df: OHLCV DataFrame.
            period: Williams %R period.
            
        Returns:
            Series with Williams %R values.
        """
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        return williams_r
    
    def commodity_channel_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            df: OHLCV DataFrame.
            period: CCI period.
            
        Returns:
            Series with CCI values.
        """
        # Calculate typical price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of typical price
        tp_sma = tp.rolling(window=period).mean()
        
        # Calculate mean deviation
        mean_deviation = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        # Calculate CCI
        # Use a small constant to avoid division by zero
        cci = (tp - tp_sma) / (0.015 * mean_deviation)
        
        return cci
    
    def money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            df: OHLCV DataFrame.
            period: MFI period.
            
        Returns:
            Series with MFI values.
        """
        # Calculate typical price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = tp * df['volume']
        
        # Get money flow direction
        direction = np.sign(tp.diff())
        
        # Separate positive and negative money flows
        positive_flow = pd.Series(
            np.where(direction > 0, raw_money_flow, 0),
            index=df.index
        )
        negative_flow = pd.Series(
            np.where(direction < 0, raw_money_flow, 0),
            index=df.index
        )
        
        # Calculate sum of positive and negative flows over period
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Calculate money flow ratio
        mf_ratio = positive_mf / negative_mf
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi
    
    def true_strength_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate True Strength Index (TSI).
        
        Args:
            df: OHLCV DataFrame.
            period: First smoothing period. Second period will be half of this.
            
        Returns:
            Series with TSI values.
        """
        # TSI uses two EMA smoothing periods
        long_period = period
        short_period = period // 2
        
        # Calculate price changes
        price_change = df['close'].diff()
        
        # Double smoothing of price changes
        first_smooth = price_change.ewm(span=long_period, adjust=False).mean()
        double_smooth = first_smooth.ewm(span=short_period, adjust=False).mean()
        
        # Double smoothing of absolute price changes
        abs_first_smooth = price_change.abs().ewm(span=long_period, adjust=False).mean()
        abs_double_smooth = abs_first_smooth.ewm(span=short_period, adjust=False).mean()
        
        # Calculate TSI
        tsi = 100 * (double_smooth / abs_double_smooth)
        
        return tsi
    
    def ultimate_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Ultimate Oscillator.
        
        Args:
            df: OHLCV DataFrame.
            period: First period. Second period will be 2x this, third period 4x.
            
        Returns:
            Series with Ultimate Oscillator values.
        """
        # Ultimate Oscillator uses three periods
        period1 = period
        period2 = period * 2
        period3 = period * 4
        
        # Calculate buying pressure (BP) and true range (TR)
        close_prev = df['close'].shift(1)
        true_low = pd.concat([df['low'], close_prev], axis=1).min(axis=1)
        
        buying_pressure = df['close'] - true_low
        true_range = self.average_true_range(df, 1)  # ATR with period 1 is just TR
        
        # Calculate averages for each period
        avg1 = buying_pressure.rolling(window=period1).sum() / true_range.rolling(window=period1).sum()
        avg2 = buying_pressure.rolling(window=period2).sum() / true_range.rolling(window=period2).sum()
        avg3 = buying_pressure.rolling(window=period3).sum() / true_range.rolling(window=period3).sum()
        
        # Calculate Ultimate Oscillator with weights
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        
        return uo
    
    #############################
    # VOLATILITY INDICATORS
    #############################
    
    def average_true_range(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: OHLCV DataFrame.
            period: ATR period.
            
        Returns:
            Series with ATR values.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        close_prev = close.shift(1)
        
        # Calculate true range
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: OHLCV DataFrame.
            period: Bollinger Bands period.
            
        Returns:
            Dictionary with middle band, upper band, and lower band.
        """
        # Standard deviation multiplier
        std_dev = 2
        
        # Calculate middle band (SMA)
        middle_band = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        }
    
    def keltner_channel(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate Keltner Channel.
        
        Args:
            df: OHLCV DataFrame.
            period: Keltner Channel period.
            
        Returns:
            Dictionary with middle line, upper line, and lower line.
        """
        # ATR multiplier
        multiplier = 2
        
        # Calculate middle line (EMA)
        middle_line = df['close'].ewm(span=period, adjust=False).mean()
        
        # Calculate ATR
        atr = self.average_true_range(df, period)
        
        # Calculate upper and lower lines
        upper_line = middle_line + (multiplier * atr)
        lower_line = middle_line - (multiplier * atr)
        
        return {
            'middle': middle_line,
            'upper': upper_line,
            'lower': lower_line
        }
    
    def donchian_channel(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate Donchian Channel.
        
        Args:
            df: OHLCV DataFrame.
            period: Donchian Channel period.
            
        Returns:
            Dictionary with middle line, upper line, and lower line.
        """
        # Calculate upper and lower lines
        upper_line = df['high'].rolling(window=period).max()
        lower_line = df['low'].rolling(window=period).min()
        
        # Calculate middle line
        middle_line = (upper_line + lower_line) / 2
        
        return {
            'middle': middle_line,
            'upper': upper_line,
            'lower': lower_line
        }
    
    def vortex_indicator(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Vortex Indicator.
        
        Args:
            df: OHLCV DataFrame.
            period: Vortex Indicator period.
            
        Returns:
            Dictionary with positive and negative Vortex Indicator values.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Shift high, low, and close
        high_prev = high.shift(1)
        low_prev = low.shift(1)
        
        # Calculate true range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate VM+
        vm_plus = (high - low_prev).abs()
        
        # Calculate VM-
        vm_minus = (low - high_prev).abs()
        
        # Sum over period
        tr_sum = true_range.rolling(window=period).sum()
        vm_plus_sum = vm_plus.rolling(window=period).sum()
        vm_minus_sum = vm_minus.rolling(window=period).sum()
        
        # Calculate VI+ and VI-
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum
        
        return {
            'plus': vi_plus,
            'minus': vi_minus
        }
    
    #############################
    # VOLUME INDICATORS
    #############################
    
    def on_balance_volume(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            df: OHLCV DataFrame.
            period: Not used, included for API consistency.
            
        Returns:
            Series with OBV values.
        """
        close = df['close']
        volume = df['volume']
        
        # Calculate price direction
        direction = np.sign(close.diff())
        
        # Initialize OBV with first volume
        obv = pd.Series(0, index=df.index)
        
        # Calculate OBV
        for i in range(1, len(df)):
            if direction.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif direction.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def volume_weighted_average_price(self, df: pd.DataFrame