"""
=============================================================================
Password History Manager
=============================================================================
Manage password history to prevent password reuse.

Security Features:
- Store last N password hashes per user
- Prevent reusing recent passwords
- Automatic password history rotation
=============================================================================
"""

from typing import List
from datetime import datetime
from passlib.context import CryptContext
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordHistoryManager:
    """
    Manage password history to prevent password reuse.

    Features:
    - Stores last 5 password hashes per user
    - Checks if new password matches any recent password
    - Automatically rotates history when limit is reached
    """

    MAX_HISTORY = 5  # Keep last 5 passwords

    @staticmethod
    def add_to_history(username: str, password_hash: str) -> bool:
        """
        Add password hash to user's password history.

        This should be called BEFORE updating the user's current password,
        to save the old password to history.

        Args:
            username: Username to add history for
            password_hash: Bcrypt hash of the OLD password to save

        Returns:
            True if successful, False otherwise
        """
        try:
            with MongoDBClient(collection_name="users") as db:
                users = db.get_collection("users")
                user = users.find_one({"username": username})

                if not user:
                    logger.warning(f"User not found for password history: {username}")
                    return False

                # Get existing history or create empty list
                history = user.get("password_history", [])

                # Add current password to front of history
                history.insert(0, password_hash)

                # Keep only last MAX_HISTORY passwords
                history = history[:PasswordHistoryManager.MAX_HISTORY]

                # Update user document
                users.update_one(
                    {"username": username},
                    {
                        "$set": {
                            "password_history": history,
                            "password_history_updated_at": datetime.utcnow()
                        }
                    }
                )

                logger.info(f"✅ Password added to history for user: {username} (history size: {len(history)})")
                return True

        except Exception as e:
            logger.error(f"Error adding password to history: {e}")
            return False

    @staticmethod
    def is_password_reused(username: str, new_password: str) -> bool:
        """
        Check if the new password matches any recent password.

        This checks both:
        1. The user's current password
        2. Passwords in the history (last 5 passwords)

        Args:
            username: Username to check
            new_password: Plain text password to verify

        Returns:
            True if password was used recently, False if it's new
        """
        try:
            with MongoDBClient(collection_name="users") as db:
                users = db.get_collection("users")
                user = users.find_one({"username": username})

                if not user:
                    logger.warning(f"User not found for password reuse check: {username}")
                    return False

                # Check against current password
                current_hash = user.get("password_hash", "")
                if current_hash and pwd_context.verify(new_password, current_hash):
                    logger.warning(f"⚠️  Password reuse detected (current password): {username}")
                    return True

                # Check against password history
                history = user.get("password_history", [])

                for old_hash in history:
                    if pwd_context.verify(new_password, old_hash):
                        logger.warning(f"⚠️  Password reuse detected (historical password): {username}")
                        return True

                logger.info(f"✅ Password is new for user: {username}")
                return False

        except Exception as e:
            logger.error(f"Error checking password reuse: {e}")
            # In case of error, allow the password change (fail open for better UX)
            return False

    @staticmethod
    def get_history_count(username: str) -> int:
        """
        Get the number of passwords in user's history.

        Args:
            username: Username to check

        Returns:
            Number of passwords in history
        """
        try:
            with MongoDBClient(collection_name="users") as db:
                users = db.get_collection("users")
                user = users.find_one({"username": username})

                if not user:
                    return 0

                history = user.get("password_history", [])
                return len(history)

        except Exception as e:
            logger.error(f"Error getting history count: {e}")
            return 0

    @staticmethod
    def clear_history(username: str) -> bool:
        """
        Clear password history for a user.

        This should only be used in special cases (e.g., user account reset).

        Args:
            username: Username to clear history for

        Returns:
            True if successful, False otherwise
        """
        try:
            with MongoDBClient(collection_name="users") as db:
                users = db.get_collection("users")

                users.update_one(
                    {"username": username},
                    {
                        "$set": {
                            "password_history": [],
                            "password_history_cleared_at": datetime.utcnow()
                        }
                    }
                )

                logger.warning(f"⚠️  Password history cleared for user: {username}")
                return True

        except Exception as e:
            logger.error(f"Error clearing password history: {e}")
            return False

    @staticmethod
    def get_last_password_change(username: str) -> str:
        """
        Get the timestamp of the last password change.

        Args:
            username: Username to check

        Returns:
            ISO format timestamp string, or "Never" if no change recorded
        """
        try:
            with MongoDBClient(collection_name="users") as db:
                users = db.get_collection("users")
                user = users.find_one({"username": username})

                if not user:
                    return "Never"

                password_changed_at = user.get("password_changed_at")

                if password_changed_at:
                    return password_changed_at.isoformat()
                else:
                    return "Never"

        except Exception as e:
            logger.error(f"Error getting last password change: {e}")
            return "Unknown"


# Example usage and testing
if __name__ == "__main__":
    print("Password History Manager - Test Mode")
    print("=" * 80)

    # Note: This requires MongoDB to be running
    # Uncomment below for testing

    # test_username = "test_user"
    # test_password_1 = "OldPassword123!"
    # test_password_2 = "NewPassword456!"
    #
    # # Test adding to history
    # old_hash = pwd_context.hash(test_password_1)
    # PasswordHistoryManager.add_to_history(test_username, old_hash)
    #
    # # Test reuse detection
    # is_reused = PasswordHistoryManager.is_password_reused(test_username, test_password_1)
    # print(f"Password reused (should be True): {is_reused}")
    #
    # is_reused_new = PasswordHistoryManager.is_password_reused(test_username, test_password_2)
    # print(f"New password reused (should be False): {is_reused_new}")
    #
    # # Test history count
    # count = PasswordHistoryManager.get_history_count(test_username)
    # print(f"History count: {count}")

    print("\n✅ Password History Manager module loaded successfully")
    print("=" * 80)
