# Combined weight
        total_weight = base_weight * accuracy_weight * account_weight
        
        # Apply multiplier and cap
        total_weight *= self.config.influencer_weight_multiplier
        return min(total_weight, 10.0)  # Cap at 10x
    
    def _get_accuracy_weight(self, username: str, platform: str) -> float:
        """Get accuracy weight based on historical performance"""
        key = f"{platform}_{username}"
        
        if key in self.influencer_db:
            influencer_data = self.influencer_db[key]
            accuracy = influencer_data.get('accuracy_score', 0.5)
            # Convert accuracy to weight (0.5 accuracy = 1.0 weight)
            return 0.5 + accuracy
        
        return 1.0  # Default weight for unknown users
    
    def update_influencer_accuracy(self, username: str, platform: str, prediction_accuracy: float):
        """Update influencer's prediction accuracy"""
        key = f"{platform}_{username}"
        
        if key not in self.influencer_db:
            self.influencer_db[key] = {
                'accuracy_score': 0.5,
                'prediction_count': 0
            }
        
        # Update with exponential moving average
        current_accuracy = self.influencer_db[key]['accuracy_score']
        prediction_count = self.influencer_db[key]['prediction_count']
        
        # Weight new accuracy more heavily for accounts with fewer predictions
        alpha = 1.0 / (prediction_count + 1)
        new_accuracy = (1 - alpha) * current_accuracy + alpha * prediction_accuracy
        
        self.influencer_db[key]['accuracy_score'] = new_accuracy
        self.influencer_db[key]['prediction_count'] += 1
    
    def get_top_influencers(self, platform: str, n: int = 10) -> List[InfluencerMetrics]:
        """Get top influencers for a platform"""
        influencers = []
        
        for key, data in self.influencer_db.items():
            if key.startswith(f"{platform}_"):
                username = key.split(f"{platform}_", 1)[1]
                influencers.append(InfluencerMetrics(
                    username=username,
                    platform=platform,
                    follower_count=data.get('follower_count', 0),
                    engagement_rate=data.get('engagement_rate', 0.0),
                    sentiment_history=data.get('sentiment_history', []),
                    accuracy_score=data.get('accuracy_score', 0.5),
                    influence_weight=self.calculate_influence_weight(
                        username, platform, data.get('follower_count', 0)
                    )
                ))
        
        # Sort by influence weight
        influencers.sort(key=lambda x: x.influence_weight, reverse=True)
        return influencers[:n]


class SocialMetricsAggregator:
    """Aggregate social data into meaningful metrics"""
    
    def __init__(self, config: SocialDataConfig):
        self.config = config
        
    async def aggregate_metrics(self, posts: List[SocialPost], timerange_hours: int = 24) -> Dict[str, SocialMetrics]:
        """Aggregate social posts into protocol-specific metrics"""
        # Group posts by protocol
        protocol_posts = defaultdict(list)
        for post in posts:
            for protocol in post.mentioned_protocols:
                protocol_posts[protocol].append(post)
        
        # Calculate metrics for each protocol
        metrics = {}
        for protocol, posts_list in protocol_posts.items():
            metrics[protocol] = await self._calculate_protocol_metrics(protocol, posts_list)
        
        return metrics
    
    async def _calculate_protocol_metrics(self, protocol: str, posts: List[SocialPost]) -> SocialMetrics:
        """Calculate metrics for a specific protocol"""
        if not posts:
            return SocialMetrics(
                protocol=protocol,
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                mention_count=0,
                unique_authors=0,
                total_engagement=0,
                influencer_sentiment=0.0,
                viral_score=0.0,
                sentiment_volatility=0.0,
                dominant_themes=[],
                risk_signals=[]
            )
        
        # Basic counts
        mention_count = len(posts)
        unique_authors = len(set(post.author for post in posts))
        
        # Sentiment analysis
        sentiments = [post.processed_sentiment for post in posts]
        confidences = [post.confidence for post in posts]
        
        # Weighted sentiment (by confidence)
        weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences))
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            sentiment_score = weighted_sentiment / total_confidence
            sentiment_confidence = total_confidence / len(posts)
        else:
            sentiment_score = 0.0
            sentiment_confidence = 0.0
        
        # Calculate sentiment volatility
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Engagement metrics
        total_engagement = 0
        for post in posts:
            engagement_sum = sum(post.engagement_metrics.values())
            total_engagement += engagement_sum
        
        # Influencer sentiment (weighted by follower count)
        influencer_sentiments = []
        influencer_weights = []
        
        for post in posts:
            if post.author_followers >= self.config.min_follower_count:
                weight = np.log10(post.author_followers / self.config.min_follower_count)
                influencer_sentiments.append(post.processed_sentiment)
                influencer_weights.append(weight)
        
        if influencer_sentiments:
            influencer_sentiment = np.average(influencer_sentiments, weights=influencer_weights)
        else:
            influencer_sentiment = sentiment_score
        
        # Viral score (based on engagement and spread)
        viral_score = self._calculate_viral_score(posts)
        
        # Extract dominant themes
        dominant_themes = self._extract_dominant_themes(posts)
        
        # Detect risk signals
        risk_signals = self._detect_risk_signals(posts)
        
        return SocialMetrics(
            protocol=protocol,
            timestamp=datetime.utcnow(),
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            mention_count=mention_count,
            unique_authors=unique_authors,
            total_engagement=total_engagement,
            influencer_sentiment=influencer_sentiment,
            viral_score=viral_score,
            sentiment_volatility=sentiment_volatility,
            dominant_themes=dominant_themes,
            risk_signals=risk_signals
        )
    
    def _calculate_viral_score(self, posts: List[SocialPost]) -> float:
        """Calculate how viral/trending a topic is"""
        if not posts:
            return 0.0
        
        # Factors for viral score
        # 1. Engagement rate
        total_engagement = sum(sum(post.engagement_metrics.values()) for post in posts)
        avg_engagement = total_engagement / len(posts)
        
        # 2. Author diversity (unique authors / total posts)
        unique_authors = len(set(post.author for post in posts))
        author_diversity = unique_authors / len(posts)
        
        # 3. Time concentration (posts in recent hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=3)
        recent_posts = sum(1 for post in posts if post.timestamp > recent_cutoff)
        time_concentration = recent_posts / len(posts)
        
        # 4. High-follower participation
        influencer_posts = sum(1 for post in posts if post.author_followers > 10000)
        influencer_participation = influencer_posts / len(posts)
        
        # Combine factors
        viral_score = (
            (avg_engagement / 1000) * 0.3 +  # Normalize engagement
            author_diversity * 0.2 +
            time_concentration * 0.3 +
            influencer_participation * 0.2
        )
        
        return min(viral_score, 1.0)  # Cap at 1.0
    
    def _extract_dominant_themes(self, posts: List[SocialPost]) -> List[str]:
        """Extract dominant themes from posts using TF-IDF"""
        # Combine all text
        all_text = " ".join(post.content for post in posts)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([all_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-10:][::-1]
            dominant_themes = [feature_names[i] for i in top_indices if scores[i] > 0.1]
            
            return dominant_themes
        except:
            # Fallback: extract from key phrases
            all_phrases = []
            for post in posts:
                if hasattr(post, 'raw_sentiment') and 'key_phrases' in post.raw_sentiment:
                    all_phrases.extend(post.raw_sentiment['key_phrases'])
            
            # Count frequencies
            phrase_counts = Counter(all_phrases)
            return [phrase for phrase, count in phrase_counts.most_common(5)]
    
    def _detect_risk_signals(self, posts: List[SocialPost]) -> List[str]:
        """Detect potential risk signals in social sentiment"""
        risk_signals = []
        
        # Check for extreme negative sentiment
        negative_posts = [post for post in posts if post.processed_sentiment < -0.5]
        if len(negative_posts) / len(posts) > 0.3:  # More than 30% very negative
            risk_signals.append("high_negative_sentiment")
        
        # Check for FUD indicators
        fud_keywords = ['scam', 'rugpull', 'exit scam', 'ponzi', 'dump', 'crash', 'hack']
        fud_posts = 0
        
        for post in posts:
            content_lower = post.content.lower()
            if any(keyword in content_lower for keyword in fud_keywords):
                fud_posts += 1
        
        if fud_posts / len(posts) > 0.1:  # More than 10% FUD posts
            risk_signals.append("fud_spreading")
        
        # Check for sudden sentiment changes
        if len(posts) >= 10:
            # Sort by timestamp
            sorted_posts = sorted(posts, key=lambda x: x.timestamp)
            
            # Split into recent and older posts
            split_point = int(len(sorted_posts) * 0.7)
            older_posts = sorted_posts[:split_point]
            recent_posts = sorted_posts[split_point:]
            
            # Calculate sentiment change
            older_sentiment = np.mean([p.processed_sentiment for p in older_posts])
            recent_sentiment = np.mean([p.processed_sentiment for p in recent_posts])
            
            sentiment_drop = older_sentiment - recent_sentiment
            if sentiment_drop > 0.3:  # Significant sentiment drop
                risk_signals.append("sentiment_deterioration")
        
        # Check for low engagement with negative sentiment
        low_engagement_negative = [
            post for post in posts 
            if post.processed_sentiment < -0.3 and sum(post.engagement_metrics.values()) < 5
        ]
        
        if len(low_engagement_negative) > len(posts) * 0.2:
            risk_signals.append("widespread_quiet_negativity")
        
        return risk_signals


class SocialDataStorage:
    """Handle storage and retrieval of social data"""
    
    def __init__(self, config: SocialDataConfig):
        self.config = config
        self.db_path = self._get_db_path()
        
    def _get_db_path(self) -> str:
        """Get database path from config"""
        db_url = self.config.database_url
        if db_url.startswith('sqlite:///'):
            return db_url.replace('sqlite:///', '')
        else:
            # For other databases, would return connection string
            return db_url
    
    async def initialize_database(self):
        """Initialize the database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS social_posts (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            author_followers INTEGER,
            timestamp DATETIME NOT NULL,
            sentiment_score REAL,
            confidence REAL,
            engagement_metrics TEXT,
            mentioned_protocols TEXT,
            metadata TEXT,
            hash TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS social_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            sentiment_score REAL,
            sentiment_confidence REAL,
            mention_count INTEGER,
            unique_authors INTEGER,
            total_engagement INTEGER,
            influencer_sentiment REAL,
            viral_score REAL,
            sentiment_volatility REAL,
            dominant_themes TEXT,
            risk_signals TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS influencers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            platform TEXT NOT NULL,
            follower_count INTEGER,
            accuracy_score REAL,
            influence_weight REAL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, platform)
        );
        
        CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON social_posts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_posts_platform ON social_posts(platform);
        CREATE INDEX IF NOT EXISTS idx_posts_mentioned ON social_posts(mentioned_protocols);
        CREATE INDEX IF NOT EXISTS idx_metrics_protocol ON social_metrics(protocol);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON social_metrics(timestamp);
        """
        
        conn = sqlite3.connect(self.db_path)
        conn.executescript(schema)
        conn.close()
    
    async def save_posts(self, posts: List[SocialPost]):
        """Save social posts to database"""
        conn = sqlite3.connect(self.db_path)
        
        for post in posts:
            # Check if post already exists
            existing = conn.execute(
                "SELECT id FROM social_posts WHERE id = ? AND platform = ?",
                (post.id, post.platform)
            ).fetchone()
            
            if existing:
                continue  # Skip duplicate
            
            # Insert post
            conn.execute("""
                INSERT INTO social_posts (
                    id, platform, content, author, author_followers, timestamp,
                    sentiment_score, confidence, engagement_metrics, mentioned_protocols,
                    metadata, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post.id,
                post.platform,
                post.content,
                post.author,
                post.author_followers,
                post.timestamp,
                post.processed_sentiment,
                post.confidence,
                json.dumps(post.engagement_metrics),
                json.dumps(post.mentioned_protocols),
                json.dumps(post.metadata),
                post.hash
            ))
        
        conn.commit()
        conn.close()
    
    async def save_metrics(self, metrics: Dict[str, SocialMetrics]):
        """Save aggregated metrics to database"""
        conn = sqlite3.connect(self.db_path)
        
        for protocol, metric in metrics.items():
            conn.execute("""
                INSERT INTO social_metrics (
                    protocol, timestamp, sentiment_score, sentiment_confidence,
                    mention_count, unique_authors, total_engagement, influencer_sentiment,
                    viral_score, sentiment_volatility, dominant_themes, risk_signals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.protocol,
                metric.timestamp,
                metric.sentiment_score,
                metric.sentiment_confidence,
                metric.mention_count,
                metric.unique_authors,
                metric.total_engagement,
                metric.influencer_sentiment,
                metric.viral_score,
                metric.sentiment_volatility,
                json.dumps(metric.dominant_themes),
                json.dumps(metric.risk_signals)
            ))
        
        conn.commit()
        conn.close()
    
    async def get_recent_posts(self, protocol: str, hours_back: int = 24) -> List[SocialPost]:
        """Retrieve recent posts for a protocol"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        cursor = conn.execute("""
            SELECT * FROM social_posts
            WHERE mentioned_protocols LIKE ?
            AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (f'%"{protocol}"%', cutoff_time))
        
        posts = []
        for row in cursor:
            post = SocialPost(
                id=row['id'],
                platform=row['platform'],
                content=row['content'],
                author=row['author'],
                author_followers=row['author_followers'],
                timestamp=row['timestamp'],
                raw_sentiment={},
                processed_sentiment=row['sentiment_score'],
                confidence=row['confidence'],
                engagement_metrics=json.loads(row['engagement_metrics']),
                mentioned_protocols=json.loads(row['mentioned_protocols']),
                metadata=json.loads(row['metadata']),
                hash=row['hash']
            )
            posts.append(post)
        
        conn.close()
        return posts
    
    async def get_metrics_history(self, protocol: str, days_back: int = 7) -> List[SocialMetrics]:
        """Get historical metrics for a protocol"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        cursor = conn.execute("""
            SELECT * FROM social_metrics
            WHERE protocol = ?
            AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (protocol, cutoff_time))
        
        metrics = []
        for row in cursor:
            metric = SocialMetrics(
                protocol=row['protocol'],
                timestamp=row['timestamp'],
                sentiment_score=row['sentiment_score'],
                sentiment_confidence=row['sentiment_confidence'],
                mention_count=row['mention_count'],
                unique_authors=row['unique_authors'],
                total_engagement=row['total_engagement'],
                influencer_sentiment=row['influencer_sentiment'],
                viral_score=row['viral_score'],
                sentiment_volatility=row['sentiment_volatility'],
                dominant_themes=json.loads(row['dominant_themes']),
                risk_signals=json.loads(row['risk_signals'])
            )
            metrics.append(metric)
        
        conn.close()
        return metrics
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage storage"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Delete old posts
        conn.execute("DELETE FROM social_posts WHERE timestamp < ?", (cutoff_time,))
        
        # Delete old metrics
        conn.execute("DELETE FROM social_metrics WHERE timestamp < ?", (cutoff_time,))
        
        conn.commit()
        conn.close()


class SocialDataOrchestrator:
    """Main orchestrator for social data pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = SocialDataConfig(**config_dict)
        else:
            self.config = SocialDataConfig()
        
        # Initialize components
        self.collectors = {}
        self.processor = SocialDataProcessor(self.config)
        self.aggregator = SocialMetricsAggregator(self.config)
        self.storage = SocialDataStorage(self.config)
        
        # Initialize collectors based on config
        if self.config.enable_twitter and self.config.twitter_bearer_token:
            self.collectors['twitter'] = TwitterCollector(self.config)
        
        if self.config.enable_reddit and self.config.reddit_client_id:
            self.collectors['reddit'] = RedditCollector(self.config)
        
        # Task queues for processing
        self.collection_queue = asyncio.Queue()
        self.processing_queue = asyncio.Queue()
        
        # Running state
        self.is_running = False
        self.tasks = []
    
    async def initialize(self):
        """Initialize the social data system"""
        logger.info("Initializing Social Data Orchestrator...")
        
        # Initialize database
        await self.storage.initialize_database()
        
        # Connect to data sources
        for name, collector in self.collectors.items():
            try:
                await collector.connect()
                logger.info(f"Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
        
        logger.info("Social Data Orchestrator initialized")
    
    async def start_continuous_collection(self, protocols: List[str]):
        """Start continuous social data collection"""
        self.is_running = True
        logger.info(f"Starting continuous collection for protocols: {protocols}")
        
        # Start collection tasks
        self.tasks.append(asyncio.create_task(
            self._collection_loop(protocols)
        ))
        
        # Start processing tasks
        self.tasks.append(asyncio.create_task(
            self._processing_loop()
        ))
        
        # Start aggregation task
        self.tasks.append(asyncio.create_task(
            self._aggregation_loop(protocols)
        ))
        
        # Start cleanup task
        self.tasks.append(asyncio.create_task(
            self._cleanup_loop()
        ))
    
    async def stop(self):
        """Stop all collection and processing tasks"""
        self.is_running = False
        logger.info("Stopping Social Data Orchestrator...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect from data sources
        for name, collector in self.collectors.items():
            try:
                await collector.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
        
        logger.info("Social Data Orchestrator stopped")
    
    async def _collection_loop(self, protocols: List[str]):
        """Main collection loop"""
        while self.is_running:
            try:
                # Collect from each enabled source
                all_posts = []
                
                for name, collector in self.collectors.items():
                    try:
                        posts = await collector.collect_data(protocols, self.config.lookback_hours)
                        all_posts.extend(posts)
                        logger.info(f"Collected {len(posts)} posts from {name}")
                    except Exception as e:
                        logger.error(f"Error collecting from {name}: {e}")
                
                # Add to processing queue
                if all_posts:
                    await self.processing_queue.put(all_posts)
                
                # Wait before next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _processing_loop(self):
        """Process collected posts"""
        while self.is_running:
            try:
                # Get posts from queue
                posts = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Process posts
                processed_posts = await self.processor.process_posts(posts)
                
                # Save processed posts
                await self.storage.save_posts(processed_posts)
                
                logger.info(f"Processed and saved {len(processed_posts)} posts")
                
            except asyncio.TimeoutError:
                continue  # No posts in queue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    async def _aggregation_loop(self, protocols: List[str]):
        """Aggregate metrics periodically"""
        while self.is_running:
            try:
                # Get recent posts for each protocol
                for protocol in protocols:
                    posts = await self.storage.get_recent_posts(protocol, 24)
                    
                    if posts:
                        # Calculate metrics
                        metrics = await self.aggregator.aggregate_metrics(posts)
                        
                        # Save metrics
                        await self.storage.save_metrics(metrics)
                        
                        logger.info(f"Aggregated metrics for {protocol}")
                
                # Wait before next aggregation (every hour)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.is_running:
            try:
                # Clean up data older than configured days
                await self.storage.cleanup_old_data(30)
                logger.info("Performed data cleanup")
                
                # Wait 24 hours before next cleanup
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def get_protocol_sentiment(self, protocol: str, hours_back: int = 24) -> SocialMetrics:
        """Get current sentiment metrics for a protocol"""
        # Try to get cached metrics first
        metrics_history = await self.storage.get_metrics_history(protocol, 1)
        
        if metrics_history and (datetime.utcnow() - metrics_history[0].timestamp).seconds < 3600:
            # Use recent cached metrics (less than 1 hour old)
            return metrics_history[0]
        
        # Calculate fresh metrics
        posts = await self.storage.get_recent_posts(protocol, hours_back)
        if posts:
            metrics = await self.aggregator.aggregate_metrics(posts)
            return metrics.get(protocol, SocialMetrics(
                protocol=protocol,
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                mention_count=0,
                unique_authors=0,
                total_engagement=0,
                influencer_sentiment=0.0,
                viral_score=0.0,
                sentiment_volatility=0.0,
                dominant_themes=[],
                risk_signals=[]
            ))
        else:
            # Return empty metrics
            return SocialMetrics(
                protocol=protocol,
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                mention_count=0,
                unique_authors=0,
                total_engagement=0,
                influencer_sentiment=0.0,
                viral_score=0.0,
                sentiment_volatility=0.0,
                dominant_themes=[],
                risk_signals=[]
            )
    
    async def get_trending_topics(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get trending topics across all protocols"""
        # This would analyze all recent posts to find trending topics
        # For now, return a simplified implementation
        all_posts = []
        
        # Get recent posts from database
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        cursor = conn.execute("""
            SELECT content, mentioned_protocols, engagement_metrics
            FROM social_posts
            WHERE timestamp >= ?
        """, (cutoff_time,))
        
        trending = []
        protocol_engagement = defaultdict(int)
        
        for row in cursor:
            protocols = json.loads(row['mentioned_protocols'])
            engagement = sum(json.loads(row['engagement_metrics']).values())
            
            for protocol in protocols:
                protocol_engagement[protocol] += engagement
        
        conn.close()
        
        # Sort by engagement
        sorted_protocols = sorted(protocol_engagement.items(), key=lambda x: x[1], reverse=True)
        
        # Get metrics for top protocols
        for protocol, engagement in sorted_protocols[:10]:
            metrics = await self.get_protocol_sentiment(protocol, hours_back)
            trending.append({
                'protocol': protocol,
                'total_engagement': engagement,
                'sentiment_score': metrics.sentiment_score,
                'viral_score': metrics.viral_score,
                'risk_signals': metrics.risk_signals
            })
        
        return trending


# Main execution functions
async def run_social_data_collection(config_path: Optional[str] = None, protocols: List[str] = None):
    """Run the social data collection system"""
    if protocols is None:
        protocols = ['ethereum', 'bitcoin', 'uniswap', 'aave', 'compound']
    
    orchestrator = SocialDataOrchestrator(config_path)
    
    try:
        await orchestrator.initialize()
        await orchestrator.start_continuous_collection(protocols)
        
        # Run indefinitely
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await orchestrator.stop()


def create_social_config(
    twitter_bearer_token: Optional[str] = None,
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None,
    protocols: List[str] = None
) -> SocialDataConfig:
    """Create a social data configuration"""
    if protocols is None:
        protocols = ['ethereum', 'bitcoin', 'uniswap', 'aave']
    
    return SocialDataConfig(
        enable_twitter=twitter_bearer_token is not None,
        enable_reddit=reddit_client_id is not None and reddit_client_secret is not None,
        twitter_bearer_token=twitter_bearer_token,
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent="MinosAI-SocialData/1.0 (by /u/MinosAI)",
        tracked_protocols=protocols,
        custom_keywords={
            'ethereum': ['eth', '$eth', 'ethereum'],
            'bitcoin': ['btc', '$btc', 'bitcoin'],
            'uniswap': ['uni', '$uni', 'uniswap'],
            'aave': ['aave', '$aave'],
            'compound': ['comp', '$comp', 'compound']
        }
    )


# Analysis and reporting functions
class SocialDataAnalyzer:
    """Advanced analysis of social data trends and patterns"""
    
    def __init__(self, storage: SocialDataStorage):
        self.storage = storage
    
    async def analyze_sentiment_trends(self, protocol: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        metrics_history = await self.storage.get_metrics_history(protocol, days_back)
        
        if not metrics_history:
            return {'error': 'No data available for analysis'}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        sentiments = [m.sentiment_score for m in metrics_history]
        volumes = [m.mention_count for m in metrics_history]
        
        # Calculate trend statistics
        sentiment_trend = self._calculate_trend(sentiments)
        volume_trend = self._calculate_trend(volumes)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(sentiments)
        
        # Correlation analysis
        sentiment_volume_correlation = np.corrcoef(sentiments, volumes)[0, 1] if len(sentiments) > 1 else 0
        
        # Moving averages
        ma_short = self._moving_average(sentiments, 7)  # 7-day MA
        ma_long = self._moving_average(sentiments, 30)  # 30-day MA
        
        # Volatility analysis
        volatility = np.std(sentiments)
        
        return {
            'protocol': protocol,
            'analysis_period': days_back,
            'sentiment_trend': {
                'direction': 'bullish' if sentiment_trend > 0.01 else 'bearish' if sentiment_trend < -0.01 else 'neutral',
                'strength': abs(sentiment_trend),
                'slope': sentiment_trend
            },
            'volume_trend': {
                'direction': 'increasing' if volume_trend > 0 else 'decreasing',
                'change_rate': volume_trend
            },
            'current_sentiment': sentiments[0] if sentiments else 0,
            'sentiment_range': {
                'min': min(sentiments) if sentiments else 0,
                'max': max(sentiments) if sentiments else 0,
                'mean': np.mean(sentiments) if sentiments else 0
            },
            'volatility': volatility,
            'anomalies': anomalies,
            'correlations': {
                'sentiment_volume': sentiment_volume_correlation
            },
            'moving_averages': {
                'short_term': ma_short[-1] if ma_short else 0,
                'long_term': ma_long[-1] if ma_long else 0,
                'golden_cross': ma_short[-1] > ma_long[-1] if ma_short and ma_long else False
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _detect_anomalies(self, values: List[float], threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies using z-score"""
        if len(values) < 3:
            return []
        
        z_scores = stats.zscore(values)
        anomalies = []
        
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > threshold:
                anomalies.append({
                    'index': i,
                    'value': values[i],
                    'z_score': z_score,
                    'type': 'positive' if z_score > 0 else 'negative'
                })
        
        return anomalies
    
    def _moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(values) < window:
            return []
        
        ma = []
        for i in range(window - 1, len(values)):
            ma.append(np.mean(values[i - window + 1:i + 1]))
        
        return ma
    
    async def compare_protocols(self, protocols: List[str], days_back: int = 7) -> Dict[str, Any]:
        """Compare sentiment metrics across protocols"""
        protocol_data = {}
        
        for protocol in protocols:
            metrics_history = await self.storage.get_metrics_history(protocol, days_back)
            
            if metrics_history:
                # Calculate aggregate metrics
                sentiments = [m.sentiment_score for m in metrics_history]
                volumes = [m.mention_count for m in metrics_history]
                viral_scores = [m.viral_score for m in metrics_history]
                
                protocol_data[protocol] = {
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments),
                    'total_mentions': sum(volumes),
                    'avg_viral_score': np.mean(viral_scores),
                    'risk_signals': [rs for m in metrics_history for rs in m.risk_signals]
                }
        
        # Rank protocols
        rankings = {
            'by_sentiment': sorted(
                protocol_data.items(),
                key=lambda x: x[1]['avg_sentiment'],
                reverse=True
            ),
            'by_volume': sorted(
                protocol_data.items(),
                key=lambda x: x[1]['total_mentions'],
                reverse=True
            ),
            'by_viral_score': sorted(
                protocol_data.items(),
                key=lambda x: x[1]['avg_viral_score'],
                reverse=True
            )
        }
        
        return {
            'protocols': protocol_data,
            'rankings': rankings,
            'analysis_period': days_back
        }
    
    async def identify_market_sentiment_shifts(self, hours_back: int = 48) -> Dict[str, Any]:
        """Identify significant market sentiment shifts"""
        # Get all recent metrics
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get sentiment data across all protocols
        cursor = conn.execute("""
            SELECT protocol, timestamp, sentiment_score, mention_count, viral_score
            FROM social_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff_time,))
        
        protocol_sentiments = defaultdict(list)
        
        for row in cursor:
            protocol_sentiments[row['protocol']].append({
                'timestamp': row['timestamp'],
                'sentiment': row['sentiment_score'],
                'volume': row['mention_count'],
                'viral_score': row['viral_score']
            })
        
        conn.close()
        
        shifts = []
        
        for protocol, data in protocol_sentiments.items():
            if len(data) < 5:
                continue
            
            # Sort by timestamp
            data.sort(key=lambda x: x['timestamp'])
            
            # Split into recent and older periods
            split_point = len(data) // 2
            older_period = data[:split_point]
            recent_period = data[split_point:]
            
            # Calculate average sentiments
            older_sentiment = np.mean([d['sentiment'] for d in older_period])
            recent_sentiment = np.mean([d['sentiment'] for d in recent_period])
            
            # Check for significant shift
            sentiment_change = recent_sentiment - older_sentiment
            
            if abs(sentiment_change) > 0.2:  # Threshold for significant shift
                shifts.append({
                    'protocol': protocol,
                    'shift_magnitude': sentiment_change,
                    'shift_direction': 'positive' if sentiment_change > 0 else 'negative',
                    'older_sentiment': older_sentiment,
                    'recent_sentiment': recent_sentiment,
                    'confidence': min(abs(sentiment_change) * 5, 1.0)  # Normalize to 0-1
                })
        
        # Sort by shift magnitude
        shifts.sort(key=lambda x: abs(x['shift_magnitude']), reverse=True)
        
        return {
            'sentiment_shifts': shifts,
            'analysis_period_hours': hours_back
        }


class SocialDataReporter:
    """Generate reports and visualizations for social data"""
    
    def __init__(self, analyzer: SocialDataAnalyzer):
        self.analyzer = analyzer
    
    async def generate_daily_summary(self, protocols: List[str]) -> Dict[str, Any]:
        """Generate daily summary report"""
        summary = {
            'date': datetime.utcnow().date().isoformat(),
            'protocols': {}
        }
        
        for protocol in protocols:
            # Get recent sentiment
            recent_metrics = await self.analyzer.storage.get_metrics_history(protocol, 1)
            
            if recent_metrics:
                metric = recent_metrics[0]
                
                # Get trend analysis
                trend_analysis = await self.analyzer.analyze_sentiment_trends(protocol, 7)
                
                summary['protocols'][protocol] = {
                    'current_sentiment': metric.sentiment_score,
                    'mention_count': metric.mention_count,
                    'risk_signals': metric.risk_signals,
                    'dominant_themes': metric.dominant_themes,
                    'viral_score': metric.viral_score,
                    'trend': trend_analysis['sentiment_trend']['direction'],
                    'volatility': trend_analysis['volatility']
                }
        
        # Get market-wide sentiment shifts
        shifts = await self.analyzer.identify_market_sentiment_shifts(24)
        summary['sentiment_shifts'] = shifts['sentiment_shifts'][:5]  # Top 5 shifts
        
        # Get trending topics
        orchestrator = SocialDataOrchestrator()
        trending = await orchestrator.get_trending_topics(24)
        summary['trending_topics'] = trending[:5]  # Top 5 trending
        
        return summary
    
    async def create_sentiment_dashboard_data(self, protocols: List[str]) -> Dict[str, Any]:
        """Create data for sentiment dashboard visualization"""
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'protocols': {},
            'market_overview': {},
            'alerts': []
        }
        
        # Protocol-specific data
        for protocol in protocols:
            metrics_history = await self.analyzer.storage.get_metrics_history(protocol, 7)
            
            if metrics_history:
                # Time series data for charts
                timestamps = [m.timestamp.isoformat() for m in metrics_history]
                sentiments = [m.sentiment_score for m in metrics_history]
                volumes = [m.mention_count for m in metrics_history]
                
                # Current state
                current = metrics_history[0]
                
                dashboard_data['protocols'][protocol] = {
                    'current_state': {
                        'sentiment_score': current.sentiment_score,
                        'mention_count': current.mention_count,
                        'viral_score': current.viral_score,
                        'risk_signals': current.risk_signals,
                        'dominant_themes': current.dominant_themes
                    },
                    'time_series': {
                        'timestamps': timestamps,
                        'sentiments': sentiments,
                        'volumes': volumes
                    },
                    'metrics': {
                        'avg_sentiment_7d': np.mean(sentiments),
                        'sentiment_volatility': np.std(sentiments),
                        'total_mentions_7d': sum(volumes)
                    }
                }
                
                # Check for alerts
                if current.risk_signals:
                    dashboard_data['alerts'].append({
                        'protocol': protocol,
                        'type': 'risk_signals',
                        'message': f"Risk signals detected for {protocol}: {', '.join(current.risk_signals)}",
                        'severity': 'warning'
                    })
                
                if current.sentiment_score < -0.5:
                    dashboard_data['alerts'].append({
                        'protocol': protocol,
                        'type': 'negative_sentiment',
                        'message': f"Very negative sentiment detected for {protocol}: {current.sentiment_score:.2f}",
                        'severity': 'danger'
                    })
        
        # Market overview
        comparison = await self.analyzer.compare_protocols(protocols, 1)
        dashboard_data['market_overview'] = {
            'most_positive': comparison['rankings']['by_sentiment'][0][0] if comparison['rankings']['by_sentiment'] else None,
            'most_negative': comparison['rankings']['by_sentiment'][-1][0] if comparison['rankings']['by_sentiment'] else None,
            'most_discussed': comparison['rankings']['by_volume'][0][0] if comparison['rankings']['by_volume'] else None,
            'most_viral': comparison['rankings']['by_viral_score'][0][0] if comparison['rankings']['by_viral_score'] else None
        }
        
        return dashboard_data
    
    def generate_text_report(self, daily_summary: Dict[str, Any]) -> str:
        """Generate a human-readable text report"""
        report_lines = [
            f"# Daily Social Sentiment Report - {daily_summary['date']}",
            "",
            "## Market Overview",
            ""
        ]
        
        # Market sentiment shifts
        if daily_summary.get('sentiment_shifts'):
            report_lines.extend([
                "### Significant Sentiment Shifts",
                ""
            ])
            
            for shift in daily_summary['sentiment_shifts']:
                direction = "📈" if shift['shift_direction'] == 'positive' else "📉"
                report_lines.append(
                    f"- **{shift['protocol']}** {direction} "
                    f"{shift['shift_direction'].title()} shift: "
                    f"{shift['shift_magnitude']:+.3f} "
                    f"(Confidence: {shift['confidence']:.1%})"
                )
            
            report_lines.append("")
        
        # Protocol analysis
        report_lines.extend([
            "## Protocol Analysis",
            ""
        ])
        
        for protocol, data in daily_summary['protocols'].items():
            sentiment_emoji = "😊" if data['current_sentiment'] > 0.2 else "😐" if data['current_sentiment'] > -0.2 else "😞"
            trend_emoji = "📈" if data['trend'] == 'bullish' else "📉" if data['trend'] == 'bearish' else "➡️"
            
            report_lines.extend([
                f"### {protocol.title()} {sentiment_emoji}",
                "",
                f"- **Sentiment Score**: {data['current_sentiment']:+.3f}",
                f"- **Mentions**: {data['mention_count']:,}",
                f"- **Trend**: {data['trend'].title()} {trend_emoji}",
                f"- **Viral Score**: {data['viral_score']:.3f}",
                f"- **Volatility**: {data['volatility']:.3f}",
                ""
            ])
            
            if data['risk_signals']:
                report_lines.extend([
                    f"⚠️ **Risk Signals**: {', '.join(data['risk_signals'])}",
                    ""
                ])
            
            if data['dominant_themes']:
                report_lines.extend([
                    f"**Key Themes**: {', '.join(data['dominant_themes'])}",
                    ""
                ])
        
        # Trending topics
        if daily_summary.get('trending_topics'):
            report_lines.extend([
                "## Trending Topics 🔥",
                ""
            ])
            
            for i, topic in enumerate(daily_summary['trending_topics'], 1):
                report_lines.append(
                    f"{i}. **{topic['protocol']}** - "
                    f"Sentiment: {topic['sentiment_score']:+.3f}, "
                    f"Engagement: {topic['total_engagement']:,}"
                )
            
            report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---",
            f"*Report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            "*Powered by Minos-AI Social Sentiment Analysis*"
        ])
        
        return "\n".join(report_lines)


# Utility functions for external integration
class SocialDataAPI:
    """API interface for external access to social data"""
    
    def __init__(self, orchestrator: SocialDataOrchestrator):
        self.orchestrator = orchestrator
        self.analyzer = SocialDataAnalyzer(orchestrator.storage)
        self.reporter = SocialDataReporter(self.analyzer)
    
    async def get_protocol_sentiment(self, protocol: str, format: str = 'json') -> Union[Dict, str]:
        """Get sentiment data for a specific protocol"""
        metrics = await self.orchestrator.get_protocol_sentiment(protocol)
        
        if format == 'json':
            return asdict(metrics)
        elif format == 'summary':
            return {
                'protocol': protocol,
                'sentiment': 'positive' if metrics.sentiment_score > 0.1 else 'negative' if metrics.sentiment_score < -0.1 else 'neutral',
                'score': metrics.sentiment_score,
                'confidence': metrics.sentiment_confidence,
                'mentions': metrics.mention_count,
                'risk_level': 'high' if metrics.risk_signals else 'normal'
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def get_market_sentiment(self, protocols: List[str] = None) -> Dict[str, Any]:
        """Get overall market sentiment"""
        if protocols is None:
            protocols = ['ethereum', 'bitcoin', 'uniswap', 'aave', 'compound']
        
        market_data = {
            'overall_sentiment': 0.0,
            'protocols': {},
            'top_risks': [],
            'trending': []
        }
        
        sentiment_scores = []
        all_risks = []
        
        for protocol in protocols:
            metrics = await self.orchestrator.get_protocol_sentiment(protocol)
            market_data['protocols'][protocol] = {
                'sentiment': metrics.sentiment_score,
                'mentions': metrics.mention_count,
                'viral_score': metrics.viral_score
            }
            
            sentiment_scores.append(metrics.sentiment_score)
            all_risks.extend([(protocol, risk) for risk in metrics.risk_signals])
        
        # Calculate overall sentiment
        if sentiment_scores:
            market_data['overall_sentiment'] = np.mean(sentiment_scores)
        
        # Get top risks
        risk_counts = Counter(all_risks)
        market_data['top_risks'] = [
            {'protocol': protocol, 'risk': risk, 'frequency': count}
            for (protocol, risk), count in risk_counts.most_common(5)
        ]
        
        # Get trending topics
        market_data['trending'] = await self.orchestrator.get_trending_topics(24)
        
        return market_data
    
    async def get_sentiment_alerts(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get alerts for extreme sentiment conditions"""
        alerts = []
        
        # Get recent sentiment shifts
        shifts = await self.analyzer.identify_market_sentiment_shifts(24)
        
        for shift in shifts['sentiment_shifts']:
            if abs(shift['shift_magnitude']) > threshold:
                alerts.append({
                    'type': 'sentiment_shift',
                    'protocol': shift['protocol'],
                    'magnitude': shift['shift_magnitude'],
                    'direction': shift['shift_direction'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'severity': 'high' if abs(shift['shift_magnitude']) > 0.7 else 'medium'
                })
        
        return alerts
    
    async def create_custom_report(self, protocols: List[str], period_days: int = 7) -> Dict[str, Any]:
        """Create a custom sentiment report"""
        report = {
            'protocols': protocols,
            'period_days': period_days,
            'generated_at': datetime.utcnow().isoformat(),
            'analysis': {}
        }
        
        for protocol in protocols:
            # Get trend analysis
            trend_analysis = await self.analyzer.analyze_sentiment_trends(protocol, period_days)
            
            # Get recent metrics
            metrics = await self.orchestrator.get_protocol_sentiment(protocol)
            
            report['analysis'][protocol] = {
                'current_sentiment': metrics.sentiment_score,
                'trend': trend_analysis['sentiment_trend'],
                'volatility': trend_analysis['volatility'],
                'anomalies': trend_analysis['anomalies'],
                'risk_signals': metrics.risk_signals,
                'key_themes': metrics.dominant_themes
            }
        
        # Get protocol comparison
        comparison = await self.analyzer.compare_protocols(protocols, period_days)
        report['comparison'] = comparison
        
        return report


# Export main classes and functions
__all__ = [
    'SocialDataConfig',
    'SocialPost',
    'SentimentAnalysis',
    'SocialMetrics',
    'InfluencerMetrics',
    'SentimentAnalyzer',
    'SocialDataCollector',
    'TwitterCollector',
    'RedditCollector',
    'SocialDataProcessor',
    'InfluencerTracker',
    'SocialMetricsAggregator',
    'SocialDataStorage',
    'SocialDataOrchestrator',
    'SocialDataAnalyzer',
    'SocialDataReporter',
    'SocialDataAPI',
    'run_social_data_collection',
    'create_social_config'
]