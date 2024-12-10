
## Tasks
### Non defined tasks

If the task is not specified and the Internet is not needed, no user feedback is checked.
This can leas to undefined task start and goal entries:

```
Sending to:  Distance Duration Team
{'task': 'Gradually increase the duration of the sit from the current time to the desired duration in small increments.', 'behavior': 'sit', 'mode': 'duration', 'status': 'current time', 'goal': 'desired duration'}
```

THis will crash the training plan generation.