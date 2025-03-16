import json
from datetime import datetime


event_log_path = "eventLogs-app-20250219140302-0000/app-20250219140302-0000"
executor_add_times = {} # executor_id -> timestamp added
executor_remove_times = {} # executor_id -> timestamp removed
task_start_times = {} # task_id -> timestamp started (optional)
executor_usage = {} # executor_id -> list of (start, end) durations
job_start_time = None # Initialize job_start_time

with open(event_log_path, "r") as f: 
    for line in f:
        try:
        
            event = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        event_type = event.get("Event", "")
    
        if event_type == "SparkListenerJobStart":
            # Record job start time (assume first job marks the application start)
            if job_start_time is None:
                job_start_time = event.get("Submission Time")
                
        elif event_type == "SparkListenerExecutorAdded":
            exec_id = event.get("Executor ID")
            time_added = event.get("Timestamp")  # Epoch millis
            executor_add_times[exec_id] = time_added
            
        elif event_type == "SparkListenerExecutorRemoved":
            exec_id = event.get("Executor ID")
            time_removed = event.get("Timestamp")
            executor_remove_times[exec_id] = time_removed

        elif event_type == "SparkListenerTaskStart":
            task_info = event.get("Task Info", {})
            task_id = task_info.get("Task ID")
            executor_id = task_info.get("Executor ID")
            start_time = task_info.get("Launch Time", 0)
            task_start_times[task_id] = (start_time, executor_id)
            
        elif event_type == "SparkListenerTaskEnd":
            task_info = event.get("Task Info", {})
            task_id = task_info.get("Task ID")
            end_time = task_info.get("Finish Time", 0)
            # Link the task to an executor and record the duration of activity
            if task_id in task_start_times:
                start_time, executor_id = task_start_times.pop(task_id)
                # Record this task execution on the executor
                executor_usage.setdefault(executor_id, []).append((start_time, end_time))
                