from prometheus_client import Counter

# Custom business metrics for the Diabetes API

# Counts total predictions, labeled by result type
PREDICTION_COUNTER = Counter(
	"diabetes_predictions_total",
	"Total number of diabetes predictions",
	["result"],
)


def inc_prediction(result: str) -> None:
	"""Increment prediction counter for a given result label.

	Args:
		result: "Diabetic" or "Non-Diabetic" (or other classification labels)
	"""
	try:
		PREDICTION_COUNTER.labels(result=result).inc()
	except Exception:
		# Metrics must never break request handling
		pass

