import asyncio
import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from email_service import email_service
from inference import SuperannuationInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.inference_engine = None

    async def initialize(self):
        """Initialize the scheduler and load ML models"""
        try:
            self.inference_engine = SuperannuationInference()
            logger.info("Scheduler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing scheduler: {e}")

    async def send_scheduled_emails(self):
        """Send scheduled financial update emails to all users"""
        try:
            logger.info("Starting scheduled email job...")

            if not self.inference_engine:
                logger.error("Inference engine not initialized")
                return

            # Get all users from the dataset
            users_df = self.inference_engine.df

            # Send admin email with general update
            admin_success = email_service.send_financial_update(is_admin=True)
            if admin_success:
                logger.info("Admin email sent successfully")
            else:
                logger.error("Failed to send admin email")

            # Send personalized emails to users (if they have email addresses)
            # Note: The current dataset doesn't have email addresses, so we'll send to admin
            # In a real implementation, you'd have user email addresses in the database

            # For demonstration, send a few sample personalized emails
            sample_users = users_df.head(3)  # Send to first 3 users as example

            for _, user in sample_users.iterrows():
                user_data = {
                    "name": user.get("Name", "Valued Client"),
                    "user_id": user.get("User_ID", "Unknown"),
                    "age": user.get("Age", 0),
                    "risk_tolerance": user.get("Risk_Tolerance", "Unknown"),
                    "annual_income": user.get("Annual_Income", 0),
                    "current_savings": user.get("Current_Savings", 0),
                }

                # Send to admin email with user-specific data
                # In production, this would be sent to the user's actual email
                user_success = email_service.send_user_specific_update(
                    os.getenv("ADMIN_EMAIL", ""), user_data
                )

                if user_success:
                    logger.info(f"User-specific email sent for {user_data['name']}")
                else:
                    logger.error(
                        f"Failed to send user-specific email for {user_data['name']}"
                    )

            logger.info("Scheduled email job completed")

        except Exception as e:
            logger.error(f"Error in scheduled email job: {e}")

    def start_scheduler(self):
        """Start the email scheduler"""
        try:
            # Schedule email job to run every 6 hours
            self.scheduler.add_job(
                self.send_scheduled_emails,
                trigger=CronTrigger(hour="*/6"),  # Every 6 hours
                id="financial_update_emails",
                name="Send Financial Update Emails",
                replace_existing=True,
            )

            # Start the scheduler
            self.scheduler.start()
            logger.info("Email scheduler started - emails will be sent every 6 hours")

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")

    def stop_scheduler(self):
        """Stop the email scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Email scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    def trigger_manual_email(
        self, user_email: str = None, user_data: Dict = None
    ) -> bool:
        """Manually trigger email sending (for testing or on-demand)"""
        try:
            if user_email and user_data:
                return email_service.send_user_specific_update(user_email, user_data)
            else:
                return email_service.send_financial_update(is_admin=True)
        except Exception as e:
            logger.error(f"Error in manual email trigger: {e}")
            return False


# Global scheduler instance
email_scheduler = EmailScheduler()
