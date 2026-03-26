"""
=============================================================================
Password Validation Module
=============================================================================
Advanced password validation with complexity requirements for Desoutter Assistant.

Security Features:
- Minimum 12 character length
- Uppercase, lowercase, digit, special character requirements
- Common password blacklist
- Password strength scoring (0-100)
=============================================================================
"""

import re
from typing import Tuple, List


class PasswordValidator:
    """
    Advanced password validation with complexity requirements.

    Requirements:
    - Minimum 12 characters
    - At least 1 uppercase letter (A-Z)
    - At least 1 lowercase letter (a-z)
    - At least 1 digit (0-9)
    - At least 1 special character (!@#$%^&*(),.?":{}|<>)
    - Not in common password list
    """

    MIN_LENGTH = 12

    # Common passwords to block
    COMMON_PASSWORDS = [
        "password", "password123", "password1234",
        "admin", "admin123", "admin1234",
        "123456", "12345678", "123456789",
        "qwerty", "qwerty123",
        "desoutter", "desoutter123",
        "tech123", "technician",
        "welcome", "welcome123",
        "letmein", "letmein123"
    ]

    @staticmethod
    def validate(password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against complexity requirements.

        Args:
            password: Password to validate

        Returns:
            (is_valid, error_messages)

        Example:
            >>> is_valid, errors = PasswordValidator.validate("Weak123")
            >>> print(is_valid, errors)
            False, ['Minimum 12 characters required', 'At least 1 special character required']
        """
        errors = []

        # Length check
        if len(password) < PasswordValidator.MIN_LENGTH:
            errors.append(f"Minimum {PasswordValidator.MIN_LENGTH} characters required")

        # Uppercase letter check
        if not re.search(r'[A-Z]', password):
            errors.append("At least 1 uppercase letter (A-Z) required")

        # Lowercase letter check
        if not re.search(r'[a-z]', password):
            errors.append("At least 1 lowercase letter (a-z) required")

        # Digit check
        if not re.search(r'\d', password):
            errors.append("At least 1 digit (0-9) required")

        # Special character check
        if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/~`]', password):
            errors.append("At least 1 special character (!@#$%^&* etc.) required")

        # Common password check
        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            errors.append("This password is too common. Please choose a more unique password.")

        # Sequential characters check (optional - disabled for better UX)
        # if re.search(r'(012|123|234|345|456|567|678|789|abc|bcd|cde|def)', password.lower()):
        #     errors.append("Avoid sequential characters (123, abc, etc.)")

        # Repeated characters check (more than 3 times)
        if re.search(r'(.)\1{3,}', password):
            errors.append("Avoid repeating the same character more than 3 times")

        return (len(errors) == 0, errors)

    @staticmethod
    def strength_score(password: str) -> int:
        """
        Calculate password strength score (0-100).

        Args:
            password: Password to evaluate

        Returns:
            Strength score from 0 (very weak) to 100 (very strong)

        Scoring:
        - Length >= 12: +25 points
        - Length >= 16: +10 points (bonus)
        - Length >= 20: +5 points (extra bonus)
        - Has uppercase: +15 points
        - Has lowercase: +15 points
        - Has digits: +15 points
        - Has special chars: +20 points
        """
        score = 0

        # Length scoring
        if len(password) >= 12:
            score += 25
        if len(password) >= 16:
            score += 10
        if len(password) >= 20:
            score += 5

        # Character type scoring
        if re.search(r'[A-Z]', password):
            score += 15
        if re.search(r'[a-z]', password):
            score += 15
        if re.search(r'\d', password):
            score += 15
        if re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/~`]', password):
            score += 20

        # Penalty for common patterns
        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            score = max(0, score - 50)

        return min(score, 100)

    @staticmethod
    def strength_label(score: int) -> str:
        """
        Get human-readable strength label.

        Args:
            score: Password strength score (0-100)

        Returns:
            Strength label: "Very Weak", "Weak", "Fair", "Strong", "Very Strong"
        """
        if score < 40:
            return "Very Weak"
        elif score < 60:
            return "Weak"
        elif score < 75:
            return "Fair"
        elif score < 90:
            return "Strong"
        else:
            return "Very Strong"

    @staticmethod
    def get_suggestions(password: str) -> List[str]:
        """
        Get suggestions to improve password strength.

        Args:
            password: Password to evaluate

        Returns:
            List of suggestions to improve the password
        """
        suggestions = []

        if len(password) < 16:
            suggestions.append("Consider using at least 16 characters for better security")

        if not re.search(r'[A-Z]', password):
            suggestions.append("Add uppercase letters")

        if not re.search(r'[a-z]', password):
            suggestions.append("Add lowercase letters")

        if not re.search(r'\d', password):
            suggestions.append("Add numbers")

        if not re.search(r'[!@#$%^&*]', password):
            suggestions.append("Add special characters")

        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            suggestions.append("Avoid common passwords")

        return suggestions


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_passwords = [
        "weak",                          # Too short
        "WeakPassword",                  # No digit, no special char
        "Weak123",                       # Too short, no special char
        "WeakPass123!",                  # Valid but short
        "StrongP@ssw0rd!",              # Valid and strong
        "VeryStr0ng!P@ssw0rd2024",      # Very strong
        "admin123",                      # Common password
    ]

    print("Password Validation Test Results:")
    print("=" * 80)

    for pwd in test_passwords:
        is_valid, errors = PasswordValidator.validate(pwd)
        score = PasswordValidator.strength_score(pwd)
        label = PasswordValidator.strength_label(score)

        print(f"\nPassword: {pwd}")
        print(f"Valid: {is_valid}")
        print(f"Strength: {score}/100 ({label})")

        if not is_valid:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")

        suggestions = PasswordValidator.get_suggestions(pwd)
        if suggestions:
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

    print("\n" + "=" * 80)
