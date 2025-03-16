import json

event_log_path = "eventLogs-app-20250219140302-0000/app-20250219140302-0000"

with open(event_log_path, "r") as f:
    for line in f:
        try:
            event = json.loads(line.strip())
            # Process the event based on its type
            event_type = event.get("Event")
            if event_type == "SparkListenerJobStart":
                # Extract job details
                job_id = event.get("Job ID")
                submission_time = event.get("Submission Time")
                # … further processing
            elif event_type == "SparkListenerStageCompleted":
                stage_info = event.get("Stage Info", {})
                stage_id = stage_info.get("Stage ID")
            # You can also get metrics from the Stage Info here.
            elif event_type == "SparkListenerTaskEnd":
                task_metrics = event.get("Task Metrics", {})
            # metrics like shuffleReadMetrics, shuffleWriteMetrics, etc.
            # … other event types as needed
        except json.JSONDecodeError:
        # Handle any lines that aren’t valid JSON
            continue
    
    print("Finished processing event log")
    print("Job details:")
    print(f"Job ID: {job_id}")


    