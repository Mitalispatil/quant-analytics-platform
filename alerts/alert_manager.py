"""
Alert manager for custom rule-based alerts
"""

from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertRule:
    """Represents a single alert rule"""
    
    def __init__(self, name: str, condition: str, threshold: float, 
                 symbol: str = None, enabled: bool = True):
        """
        Initialize alert rule
        
        Args:
            name: Name of the alert
            condition: Condition type ('zscore_gt', 'zscore_lt', 'spread_gt', 'spread_lt', 'price_gt', 'price_lt')
            threshold: Threshold value
            symbol: Symbol to monitor (None for all)
            enabled: Whether the alert is enabled
        """
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.symbol = symbol
        self.enabled = enabled
        self.triggered = False
        self.last_triggered = None
    
    def check(self, value: float, symbol: str = None) -> bool:
        """
        Check if alert condition is met
        
        Args:
            value: Current value to check
            symbol: Symbol being checked
        
        Returns:
            True if condition is met
        """
        if not self.enabled:
            return False
        
        if self.symbol and symbol and self.symbol != symbol:
            return False
        
        if value is None or (isinstance(value, float) and (value != value)):  # Check for NaN
            return False
        
        triggered = False
        
        if self.condition == 'zscore_gt':
            triggered = value > self.threshold
        elif self.condition == 'zscore_lt':
            triggered = value < -self.threshold
        elif self.condition == 'spread_gt':
            triggered = value > self.threshold
        elif self.condition == 'spread_lt':
            triggered = value < -self.threshold
        elif self.condition == 'price_gt':
            triggered = value > self.threshold
        elif self.condition == 'price_lt':
            triggered = value < self.threshold
        else:
            logger.warning(f"Unknown condition type: {self.condition}")
            return False
        
        if triggered and not self.triggered:
            # Alert just triggered
            self.triggered = True
            self.last_triggered = datetime.now()
            return True
        elif not triggered:
            self.triggered = False
        
        return False


class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.callbacks: List[Callable] = []
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove an alert rule by name"""
        self.rules = [r for r in self.rules if r.name != name]
    
    def add_callback(self, callback: Callable):
        """Add a callback function to be called when alerts trigger"""
        self.callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Dict], symbol: str = None):
        """
        Check all alert rules against current metrics
        
        Args:
            metrics: Dictionary of metric values, e.g., {'zscore': 2.5, 'spread': 100}
            symbol: Symbol being checked
        """
        triggered_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Determine which metric to check based on condition
            if 'zscore' in rule.condition:
                value = metrics.get('zscore')
            elif 'spread' in rule.condition:
                value = metrics.get('spread')
            elif 'price' in rule.condition:
                value = metrics.get('price')
            else:
                continue
            
            if rule.check(value, symbol):
                triggered_alerts.append({
                    'rule': rule,
                    'value': value,
                    'symbol': symbol
                })
        
        # Call callbacks for triggered alerts
        for alert in triggered_alerts:
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        return triggered_alerts
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        return self.rules
    
    def get_enabled_rules(self) -> List[AlertRule]:
        """Get all enabled alert rules"""
        return [r for r in self.rules if r.enabled]

