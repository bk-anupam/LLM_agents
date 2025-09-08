import argparse
import asyncio
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.firestore_thread_manager import FirestoreThreadManager


async def main():
    """
    Main async function to handle thread management from the command line.
    Usage: 1. List threads for a user -> python -m RAG_BOT.src.telegram.thread_cli list <USER_ID>        
           2. Delete a specific thread for a user -> python -m RAG_BOT.src.telegram.thread_cli delete <USER_ID> <THREAD_ID>
    """
    parser = argparse.ArgumentParser(
        description="CLI utility to manage user conversation threads from the backend.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- List command ---
    list_parser = subparsers.add_parser("list", help="List all conversation threads for a user.")
    list_parser.add_argument("user_id", type=str, help="The ID of the user whose threads to list.")

    # --- Delete command ---
    delete_parser = subparsers.add_parser("delete", help="Delete a specific conversation thread for a user.")
    delete_parser.add_argument("user_id", type=str, help="The ID of the user who owns the thread.")
    delete_parser.add_argument("thread_id", type=str, help="The ID of the thread to delete.")

    args = parser.parse_args()

    try:
        config = Config()
        thread_manager = FirestoreThreadManager(project_id=config.GCP_PROJECT_ID)

        if args.command == "list":
            user_id = args.user_id
            logger.info(f"Listing threads for user '{user_id}'...")
            threads = await thread_manager.list_threads(user_id)
            if not threads:
                print(f"No threads found for user {user_id}.")
                return

            print(f"\n--- Threads for User: {user_id} ---")
            for i, thread in enumerate(threads, 1):
                status = "[ACTIVE]" if thread.is_active else "[Archived]" if thread.is_archived else ""
                last_modified_str = thread.last_modified_at.strftime('%Y-%m-%d %H:%M UTC') if thread.last_modified_at else "N/A"
                print(f"{i}. {status} Title: '{thread.title}'")
                print(f"   Thread ID: {thread.thread_id}")
                print(f"   Last Active: {last_modified_str}")
                print("-" * 20)
            print("--- End of List ---")

        elif args.command == "delete":
            user_id = args.user_id
            thread_id = args.thread_id
            # Confirmation prompt for safety
            confirm = input(f"Are you sure you want to PERMANENTLY DELETE thread '{thread_id}' for user '{user_id}'? This cannot be undone. (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Deletion cancelled by user.")
                return
            
            success = await thread_manager.delete_thread(user_id, thread_id)
            if success:                
                print(f"Successfully deleted thread '{thread_id}'.")
            else:                
                print(f"Failed to delete thread '{thread_id}'. See logs for more information.")

    except Exception as e:
        logger.critical(f"An error occurred in the thread CLI utility: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())