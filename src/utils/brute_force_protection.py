"""
=============================================================================
Brute Force Protection System
=============================================================================
Protects login endpoints from brute force attacks with:
- IP-based tracking
- Username-based tracking
- Progressive ban system (5/10/15 attempts)
- Automatic cleanup of old records
- MongoDB integration

Usage:
    from src.utils.brute_force_protection import BruteForceProtection

    # Check if blocked
    result = BruteForceProtection.check_and_block(username, ip)
    if result["blocked"]:
        raise HTTPException(429, detail=result["message"])

    # Record failed attempt
    BruteForceProtection.record_failed_attempt(username, ip)

    # Clear on successful login
    BruteForceProtection.clear_attempts(username, ip)
=============================================================================
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Configuration from environment variables
BRUTE_FORCE_ENABLED = os.getenv("BRUTE_FORCE_ENABLED", "true").lower() == "true"

# Progressive ban levels
MAX_ATTEMPTS_LEVEL_1 = int(os.getenv("BRUTE_FORCE_MAX_ATTEMPTS_LEVEL_1", "5"))    # 5 attempts
MAX_ATTEMPTS_LEVEL_2 = int(os.getenv("BRUTE_FORCE_MAX_ATTEMPTS_LEVEL_2", "10"))   # 10 attempts
MAX_ATTEMPTS_LEVEL_3 = int(os.getenv("BRUTE_FORCE_MAX_ATTEMPTS_LEVEL_3", "15"))   # 15 attempts

# Ban durations in minutes
BAN_DURATION_LEVEL_1 = int(os.getenv("BRUTE_FORCE_BAN_DURATION_1", "15"))    # 15 minutes
BAN_DURATION_LEVEL_2 = int(os.getenv("BRUTE_FORCE_BAN_DURATION_2", "60"))    # 1 hour
BAN_DURATION_LEVEL_3 = int(os.getenv("BRUTE_FORCE_BAN_DURATION_3", "1440"))  # 24 hours


class BruteForceProtection:
    """Brute force attack protection system"""

    COLLECTION_NAME = "failed_login_attempts"

    @staticmethod
    def _get_collection():
        """Get MongoDB collection for failed attempts"""
        db = MongoDBClient()
        return db.get_collection(BruteForceProtection.COLLECTION_NAME)

    @staticmethod
    def _calculate_ban_duration(attempts: int) -> int:
        """
        Calculate ban duration based on attempt count.

        Args:
            attempts: Number of failed attempts

        Returns:
            Ban duration in minutes
        """
        if attempts >= MAX_ATTEMPTS_LEVEL_3:
            return BAN_DURATION_LEVEL_3  # 24 hours
        elif attempts >= MAX_ATTEMPTS_LEVEL_2:
            return BAN_DURATION_LEVEL_2  # 1 hour
        elif attempts >= MAX_ATTEMPTS_LEVEL_1:
            return BAN_DURATION_LEVEL_1  # 15 minutes
        return 0

    @staticmethod
    def _get_ban_level(attempts: int) -> int:
        """Get ban level (1, 2, or 3)"""
        if attempts >= MAX_ATTEMPTS_LEVEL_3:
            return 3
        elif attempts >= MAX_ATTEMPTS_LEVEL_2:
            return 2
        elif attempts >= MAX_ATTEMPTS_LEVEL_1:
            return 1
        return 0

    @staticmethod
    def check_and_block(username: str, ip_address: str) -> Dict:
        """
        Check if username/IP is blocked due to failed attempts.

        Args:
            username: Username attempting to login
            ip_address: IP address of the request

        Returns:
            {
                "blocked": bool,
                "message": str,
                "retry_after": int (minutes),
                "attempts": int
            }
        """
        if not BRUTE_FORCE_ENABLED:
            return {"blocked": False, "message": "", "retry_after": 0, "attempts": 0}

        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)
                now = datetime.utcnow()

                # Check for existing record
                record = collection.find_one({
                    "username": username.lower(),
                    "ip_address": ip_address
                })

                if not record:
                    return {"blocked": False, "message": "", "retry_after": 0, "attempts": 0}

                # Check if currently blocked
                blocked_until = record.get("blocked_until")
                if blocked_until and blocked_until > now:
                    retry_minutes = int((blocked_until - now).total_seconds() / 60) + 1
                    logger.warning(
                        f"🚫 Blocked login attempt - Username: {username}, IP: {ip_address}, "
                        f"Attempts: {record['attempts']}, Retry in: {retry_minutes} min"
                    )
                    return {
                        "blocked": True,
                        "message": f"Too many failed attempts. Try again in {retry_minutes} minutes.",
                        "retry_after": retry_minutes,
                        "attempts": record['attempts']
                    }

                return {"blocked": False, "message": "", "retry_after": 0, "attempts": record.get('attempts', 0)}

        except Exception as e:
            logger.error(f"Error checking brute force protection: {e}")
            # Fail open - don't block on error
            return {"blocked": False, "message": "", "retry_after": 0, "attempts": 0}

    @staticmethod
    def record_failed_attempt(username: str, ip_address: str) -> None:
        """
        Record a failed login attempt and apply ban if threshold reached.

        Args:
            username: Username that failed login
            ip_address: IP address of the request
        """
        if not BRUTE_FORCE_ENABLED:
            return

        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)
                now = datetime.utcnow()
                username_lower = username.lower()

                # Find existing record
                record = collection.find_one({
                    "username": username_lower,
                    "ip_address": ip_address
                })

                if record:
                    # Increment attempts
                    new_attempts = record['attempts'] + 1
                    ban_duration = BruteForceProtection._calculate_ban_duration(new_attempts)
                    ban_level = BruteForceProtection._get_ban_level(new_attempts)

                    update_data = {
                        "attempts": new_attempts,
                        "last_attempt": now
                    }

                    # Apply ban if threshold reached
                    if ban_duration > 0:
                        blocked_until = now + timedelta(minutes=ban_duration)
                        update_data["blocked_until"] = blocked_until
                        update_data["ban_level"] = ban_level

                        logger.warning(
                            f"⚠️  Ban applied - Username: {username}, IP: {ip_address}, "
                            f"Attempts: {new_attempts}, Level: {ban_level}, Duration: {ban_duration} min"
                        )

                    collection.update_one(
                        {"username": username_lower, "ip_address": ip_address},
                        {"$set": update_data}
                    )
                else:
                    # Create new record
                    collection.insert_one({
                        "username": username_lower,
                        "ip_address": ip_address,
                        "attempts": 1,
                        "first_attempt": now,
                        "last_attempt": now,
                        "blocked_until": None,
                        "ban_level": 0
                    })
                    logger.info(f"📝 First failed attempt recorded - Username: {username}, IP: {ip_address}")

        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")

    @staticmethod
    def clear_attempts(username: str, ip_address: str) -> None:
        """
        Clear failed attempts for username/IP (called on successful login).

        Args:
            username: Username that successfully logged in
            ip_address: IP address of the request
        """
        if not BRUTE_FORCE_ENABLED:
            return

        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)

                result = collection.delete_one({
                    "username": username.lower(),
                    "ip_address": ip_address
                })

                if result.deleted_count > 0:
                    logger.info(f"✅ Cleared failed attempts - Username: {username}, IP: {ip_address}")

        except Exception as e:
            logger.error(f"Error clearing attempts: {e}")

    @staticmethod
    def get_blocked_list(limit: int = 50) -> List[Dict]:
        """
        Get list of currently blocked IPs/usernames (admin only).

        Args:
            limit: Maximum number of records to return

        Returns:
            List of blocked records
        """
        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)
                now = datetime.utcnow()

                # Find currently blocked records
                blocked = list(collection.find({
                    "blocked_until": {"$gt": now}
                }).sort("blocked_until", -1).limit(limit))

                # Format for response
                result = []
                for record in blocked:
                    retry_minutes = int((record["blocked_until"] - now).total_seconds() / 60) + 1
                    result.append({
                        "username": record["username"],
                        "ip_address": record["ip_address"],
                        "attempts": record["attempts"],
                        "ban_level": record.get("ban_level", 0),
                        "blocked_until": record["blocked_until"].isoformat(),
                        "retry_after_minutes": retry_minutes,
                        "first_attempt": record.get("first_attempt", "").isoformat() if record.get("first_attempt") else None
                    })

                return result

        except Exception as e:
            logger.error(f"Error getting blocked list: {e}")
            return []

    @staticmethod
    def unblock(username: Optional[str] = None, ip_address: Optional[str] = None) -> int:
        """
        Manually unblock a username or IP (admin only).

        Args:
            username: Username to unblock (optional)
            ip_address: IP address to unblock (optional)

        Returns:
            Number of records unblocked
        """
        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)

                query = {}
                if username:
                    query["username"] = username.lower()
                if ip_address:
                    query["ip_address"] = ip_address

                if not query:
                    logger.warning("Unblock called without username or IP")
                    return 0

                result = collection.delete_many(query)

                if result.deleted_count > 0:
                    logger.info(f"🔓 Manual unblock - Username: {username}, IP: {ip_address}, Count: {result.deleted_count}")

                return result.deleted_count

        except Exception as e:
            logger.error(f"Error unblocking: {e}")
            return 0

    @staticmethod
    def cleanup_old_records(days: int = 30) -> int:
        """
        Clean up old failed attempt records (for maintenance).

        Args:
            days: Delete records older than this many days

        Returns:
            Number of records deleted
        """
        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                result = collection.delete_many({
                    "last_attempt": {"$lt": cutoff_date}
                })

                if result.deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {result.deleted_count} old brute force records (older than {days} days)")

                return result.deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0

    @staticmethod
    def get_stats() -> Dict:
        """
        Get brute force protection statistics (admin only).

        Returns:
            Statistics dictionary
        """
        try:
            with MongoDBClient() as db:
                collection = db.get_collection(BruteForceProtection.COLLECTION_NAME)
                now = datetime.utcnow()

                total_records = collection.count_documents({})
                currently_blocked = collection.count_documents({
                    "blocked_until": {"$gt": now}
                })

                # Attempts in last 24 hours
                last_24h = now - timedelta(hours=24)
                recent_attempts = collection.count_documents({
                    "last_attempt": {"$gte": last_24h}
                })

                return {
                    "enabled": BRUTE_FORCE_ENABLED,
                    "total_records": total_records,
                    "currently_blocked": currently_blocked,
                    "recent_attempts_24h": recent_attempts,
                    "config": {
                        "max_attempts_level_1": MAX_ATTEMPTS_LEVEL_1,
                        "max_attempts_level_2": MAX_ATTEMPTS_LEVEL_2,
                        "max_attempts_level_3": MAX_ATTEMPTS_LEVEL_3,
                        "ban_duration_level_1": BAN_DURATION_LEVEL_1,
                        "ban_duration_level_2": BAN_DURATION_LEVEL_2,
                        "ban_duration_level_3": BAN_DURATION_LEVEL_3
                    }
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"enabled": BRUTE_FORCE_ENABLED, "error": str(e)}
