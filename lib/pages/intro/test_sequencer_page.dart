import 'package:flutter/material.dart';
import 'rest_page.dart';

class TestSequencerPage extends StatefulWidget {
  final String participantId;
  final List<String> testOrder;

  const TestSequencerPage({
    Key? key,
    required this.participantId,
    required this.testOrder,
  }) : super(key: key);

  @override
  State<TestSequencerPage> createState() => _TestSequencerPageState();
}

class _TestSequencerPageState extends State<TestSequencerPage> {
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    // Execute after one frame (after the root build is complete).
    WidgetsBinding.instance.addPostFrameCallback((_) => _runNextStage());
  }

  /// Automatically proceeds with the test-rest-test sequence.
  Future<void> _runNextStage() async {
    // If all tests are completed, navigate to the end page.
    if (_currentIndex >= widget.testOrder.length) {
      print("All cognitive tests completed for ${widget.participantId}. Navigating to end page.");
      Navigator.pushReplacementNamed(context, '/end_page', arguments: widget.participantId);
      return;
    }

    // After the first test, there is a 2-minute rest.
    if (_currentIndex > 0) {
      print("Starting 2-minute rest before test ${_currentIndex + 1}");
      await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => const RestPage(duration: Duration(minutes: 2)),
        ),
      );
      print("Rest completed, proceeding to next test");
    }

    // Navigate to the actual test page – must be pop()'d on completion.
    final String route = widget.testOrder[_currentIndex];
    print("Starting test ${_currentIndex + 1}: $route for participant ${widget.participantId}");
    
    await Navigator.pushNamed(context, route, arguments: widget.participantId);
    
    print("Test ${_currentIndex + 1} completed: $route");
    _currentIndex++;
    
    if (mounted) {
      _runNextStage(); // Recursive call for the next stage.
    }
  }

  @override
  Widget build(BuildContext context) {
    // Empty screen + block back navigation.
    return WillPopScope(
      onWillPop: () async => false, // Block back navigation.
      child: Scaffold(
        backgroundColor: Colors.white,
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(
                "검사를 준비하고 있습니다...",
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
} 