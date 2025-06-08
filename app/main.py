# main.py (No significant changes, just for completeness)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from app.core.config import settings
from app.api.endpoints import whatsapp, prompt, admin, chat
from app.services.rss_service import update_all_feeds # Import the function

# --- SCHEDULER SETUP ---
scheduler = BackgroundScheduler()

# --- FASTAPI APP ---
app = FastAPI(title=settings.PROJECT_NAME)

@app.on_event("startup")
def startup_event():
    """
    This function runs once when the application starts.
    We use it to start our background tasks.
    """
    print("Application startup...")
    # Schedule the RSS feed update to run every 1 hour.
    # replace_existing=True is good practice to ensure only one job with this ID exists.
    scheduler.add_job(
        update_all_feeds,
        'interval',
        hours=1,
        id="update_feeds_hourly_job", # Give a distinct ID
        replace_existing=True
    )
    # Also run the job once immediately on startup.
    # Using a different ID or ensuring replace_existing=True for both
    # handles cases where startup might be called multiple times.
    scheduler.add_job(update_all_feeds, id="initial_feed_update_run", replace_existing=True)

    scheduler.start()
    print("Startup complete. RSS feed updater is running in the background.")

@app.on_event("shutdown")
def shutdown_event():
    """This function runs when the application is shutting down."""
    print("Application shutdown...")
    scheduler.shutdown()
    print("Background scheduler has been shut down.")

# Mount the 'static' directory to serve your frontend files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include all your API endpoint routers
app.include_router(whatsapp.router, prefix=settings.API_V1_STR)
app.include_router(prompt.router, prefix=settings.API_V1_STR)
app.include_router(admin.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}