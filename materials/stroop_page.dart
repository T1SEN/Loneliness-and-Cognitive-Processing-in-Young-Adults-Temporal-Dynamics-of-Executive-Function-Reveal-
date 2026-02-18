// This file implements the classic Stroop task.
// The Stroop effect is the delay in reaction time between congruent and
// incongruent stimuli.
//
// The task structure is as follows:
// 1. A word is displayed on the screen, written in a specific color.
// 2. The participant must identify the ink color of the word, ignoring the
//    word's meaning.
// 3. Trials can be:
//    - Congruent: The word's meaning and ink color are the same (e.g., "Red" in red ink).
//    - Incongruent: The word's meaning and ink color differ (e.g., "Red" in blue ink).
//    - Neutral: The word is unrelated to colors (e.g., "Train" in green ink).
// 4. The experiment includes practice trials and main trials.
// 5. Data (responses, RTs, accuracy) is logged and saved to Firestore.

// ---------------------------------------------------------------------------
// Section 0: External Packages
// ---------------------------------------------------------------------------
import 'dart:async';
import 'dart:html' as html; // For high-precision timing (window.performance.now())
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

// ---------------------------------------------------------------------------
// Section 1: Root Widget
// ---------------------------------------------------------------------------
/// The main widget for the Stroop task screen.
class StroopPage extends StatefulWidget {
  const StroopPage({Key? key}) : super(key: key);
  @override
  State<StroopPage> createState() => _StroopPageState();
}

// ---------------------------------------------------------------------------
// Section 2: Experiment Stage Enum
// ---------------------------------------------------------------------------
/// Defines the different stages of the experiment to control UI and logic flow.
enum StroopStage {
  welcome,              // Initial welcome screen.
  instructions,         // General instructions screen.
  practiceInstructions, // Instructions for the practice block.
  practiceTrial,        // An active practice trial.
  practiceFeedback,     // Feedback shown after a practice trial.
  mainTrialInstructions,// Instructions for the main experiment block.
  mainTrial,            // An active main trial.
  result,               // Final screen after experiment completion.
}

// ---------------------------------------------------------------------------
// Section 3: State Class
// ---------------------------------------------------------------------------
/// Holds the state and business logic for the entire Stroop experiment.
class _StroopPageState extends State<StroopPage> {
  // -------------------------------------------------------------------------
  // 3.1. Core State and Participant Variables
  // -------------------------------------------------------------------------
  late String _participantId;
  late final bool _reverseBtn;    // Counterbalance button order.
  StroopStage _stage = StroopStage.welcome;
  bool _initialized = false;
  bool _hasIdError = false;     // Flag to track if there was an error getting the participant ID.

  // -------------------------------------------------------------------------
  // 3.2. Practice Trial State
  // -------------------------------------------------------------------------
  late final List<Map<String, dynamic>> _practiceTrials;
  int  _pIdx = 0;                 // Current practice trial index.
  bool _pCorrect = false;         // Was the last practice response correct?

  // -------------------------------------------------------------------------
  // 3.3. Main Trial State and Stimuli
  // -------------------------------------------------------------------------
  static const List<String> _words = ["빨강", "초록", "파랑", "노랑"]; // "Red", "Green", "Blue", "Yellow"
  static const List<String> _colors = ["red", "green", "blue", "yellow"];
  static const List<String> _neutralWords = ["기차", "학교", "가방"]; // "Train", "School", "Bag"

  late final List<Map<String, dynamic>> _trials; // Holds all main trials.
  int    _tIdx = 0;               // Current main trial index.
  bool   _showFixation = false;   // Show fixation cross?
  bool   _canRespond = false;     // Is the user allowed to respond?
  String _curTxt = "";            // The current word stimulus text.
  String _curColorKey = "red";    // The current ink color stimulus.

  // -------------------------------------------------------------------------
  // 3.4. Timing and Data Logging
  // -------------------------------------------------------------------------
  DateTime? _mainTestStartTime; // Timestamp for the start of the main block.
  DateTime? _mainTestEndTime;   // Timestamp for the end of the main block.

  double? _stimStartTime;       // High-precision timestamp for stimulus onset.
  Timer? _timeout;              // Timer to handle response timeouts.

  /// A list of maps, where each map stores the data for one main trial.
  final List<Map<String, dynamic>> _trialLog = [];

  // -------------------------------------------------------------------------
  // 3.5. Static UI Mappings
  // -------------------------------------------------------------------------
  /// Maps color keys to their Korean labels for UI buttons.
  static const Map<String, String> _labelMap = {
    "red":    "빨강", // "Red"
    "green":  "초록", // "Green"
    "blue":   "파랑", // "Blue"
    "yellow": "노랑", // "Yellow"
  };

  // -------------------------------------------------------------------------
  // Section 3.6: Widget Lifecycle and Initialization
  // -------------------------------------------------------------------------
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_initialized) return;

    // Retrieve participant ID from the arguments passed to this page.
    final arguments = ModalRoute.of(context)?.settings.arguments;
    if (arguments != null) {
      final arg = arguments;
      if (arg is String) {
        _participantId = arg;
        print("StroopPage received participantId: $_participantId");
        // Counterbalance button order based on a hash of the participant ID.
        _reverseBtn = (_participantId.hashCode % 2) == 1;
      } else {
        // Handle cases where the argument is not a String.
        print("Warning: Received non-String argument for participantId. Type: ${arg.runtimeType}");
        _participantId = "unknown_participant_error";
        _hasIdError = true;
        // "Error: User ID not received."
        _showErrorAndPreventProceeding('오류: 사용자 ID를 받지 못했습니다.');
      }
    } else {
      // Handle cases where no arguments are passed at all.
      print("Warning: No arguments received for participantId.");
      _participantId = "unknown_participant_no_args";
      _hasIdError = true;
      // "Error: User ID was not passed."
      _showErrorAndPreventProceeding('오류: 사용자 ID가 전달되지 않았습니다.');
    }

    // Create a Random object based on the ID to ensure reproducible shuffling.
    final randomForShuffle = Random(_participantId.hashCode);
    
    // Initialize and shuffle practice trials, preventing immediate repeats.
    _practiceTrials = _buildPracticeTrials(randomForShuffle)..shuffle(randomForShuffle);
    _shuffleNoRepeat(_practiceTrials, randomForShuffle);

    // Initialize and shuffle main trials, preventing immediate repeats.
    _trials = _buildMainTrials()..shuffle(randomForShuffle);
    _shuffleNoRepeat(_trials, randomForShuffle);

    _initialized = true;
  }
  
  /// Helper to show a SnackBar error message and set the error flag.
  void _showErrorAndPreventProceeding(String message) {
    if (!mounted) return;
    setState(() { 
      // This flag will be used to disable the 'Next' button on the welcome screen.
      _stage = StroopStage.welcome;
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    });
  }

  @override
  void initState() {
    super.initState();
    // Trial initialization logic was moved to didChangeDependencies
    // because it requires `context` to get arguments.
  }

  @override
  void dispose() {
    // Cancel any active timers to prevent memory leaks.
    _timeout?.cancel();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Section 4: Practice Trial Logic
  // -------------------------------------------------------------------------
  /// Handles the response for a practice trial.
  void _onPracticeSelect(String colorKey) {
    final tr = _practiceTrials[_pIdx];
    _pCorrect = colorKey == tr["letterColor"];
    setState(() => _stage = StroopStage.practiceFeedback);
  }

  // -------------------------------------------------------------------------
  // Section 5: Main Trial Flow
  // -------------------------------------------------------------------------
  /// Manages the sequence of events within a single main trial.
  void _startNextTrial() async {
    if (!mounted) return;

    // Record start time at the beginning of the first main trial.
    if (_tIdx == 0 && _mainTestStartTime == null) {
      _mainTestStartTime = DateTime.now();
    }

    // If all trials are complete, move to the result screen.
    if (_tIdx >= _trials.length) {
      _mainTestEndTime = DateTime.now();
      setState(() => _stage = StroopStage.result);
      return;
    }

    // 1. Show fixation cross for 500ms.
    setState(() {
      _showFixation = true;
      _canRespond = false;
    });
    await Future.delayed(const Duration(milliseconds: 500));
    if (!mounted) return;

    // 2. Show a blank screen for 100ms (inter-stimulus interval).
    setState(() {
      _showFixation = false;
      _curTxt = ""; // Blank text induces a blank screen.
      _canRespond = false;
    });
    await Future.delayed(const Duration(milliseconds: 100));
    if (!mounted) return;

    // 3. Display the Stroop stimulus.
    final tr = _trials[_tIdx];
    setState(() {
      _curTxt = tr["text"];
      _curColorKey = tr["letterColor"];
      _canRespond = true; // Enable response buttons.
    });

    // Record the high-precision start time and set a 3-second timeout AFTER the frame is drawn.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _stimStartTime = html.window.performance.now();
      _timeout?.cancel();
      _timeout = Timer(const Duration(seconds: 3), () => _record(null)); // On timeout, record as no response.
    });
  }

  /// Handles a response from the main trial UI buttons.
  void _onMainSelect(String colorKey, PointerDownEvent event) {
    if (!_canRespond) return;
    // Use high-precision timing with performance.now() for consistency
    final double respNow = html.window.performance.now();
    final double rt = _stimStartTime == null ? 0.0 : respNow - _stimStartTime!;
    _record(colorKey, rt, respNow);
  }

  /// Records the data for the completed trial and starts the next one.
  /// [sel]: The color key selected by the user, or null for a timeout.
  /// [rt]: The calculated reaction time (ms) - null/missing for timeout.
  /// [respTimeMs]: Response time in ms (relative time) - null for timeout.
  void _record(String? sel, [double? rt, double? respTimeMs]) {
    _timeout?.cancel();
    _canRespond = false;

    final tr = _trials[_tIdx];
    final bool isTimeout = (sel == null);
    
    // Response time: use provided value or null for timeout
    final double? respMs = isTimeout ? null : (respTimeMs ?? html.window.performance.now());
    
    // RT recalculation: if both onset and response exist, calculate difference
    final double? rtMs = isTimeout
        ? null
        : ((respMs != null && _stimStartTime != null) ? (respMs - _stimStartTime!) : rt);
    
    final bool correct = sel == tr["letterColor"];

    _trialLog.add({
      // Common schema fields for Joint Bayesian
      "participant_id": _participantId,
      "task": "stroop",
      "trial_index": _tIdx,
      "block_index": null,  // Stroop doesn't have blocks
      "stim_onset_ms": _stimStartTime,
      "resp_time_ms": respMs,
      "rt_ms": rtMs,
      "correct": correct,
      "timeout": isTimeout,
      "cond": tr["type"],  // congruent/incongruent/neutral
      
      // Extra task-specific fields
      "extra": {
        "text": tr["text"],
        "letterColor": tr["letterColor"],
        "userColor": sel ?? "noResp",
      },
      
      // Compatibility fields
      "trial": _tIdx,
      ...tr,
      "userColor": sel ?? "noResp",
      "is_timeout": isTimeout,
    });

    _tIdx++;
    _startNextTrial();
  }

  // -------------------------------------------------------------------------
  // Section 6: Data Saving
  // -------------------------------------------------------------------------
  /// Calculates summary stats, packages all data, saves to Firestore, and navigates away.
  Future<void> _saveResultsAndNavigate() async {
    if (!mounted) return;
    // Show a "Saving..." dialog.
    showDialog(
      context: context, 
      barrierDismissible: false, 
      builder: (_) => const AlertDialog(
        content: Row(children: [CircularProgressIndicator(), SizedBox(width: 16), Text("저장 중...")]), // "Saving..."
      ),
    );

    try {
      // --- Calculate summary statistics ---
      final byType = <String, List<Map<String, dynamic>>>{};
      for (final t in _trialLog) {
        (byType[t["type"]] ??= []).add(t);
      }
      final totalMRT = _mrtMs(_trialLog);
      final congRT = byType["congruent"] != null ? _mrtMs(byType["congruent"]!) : null;
      final inconRT = byType["incongruent"] != null ? _mrtMs(byType["incongruent"]!) : null;
      // The Stroop Effect is the difference in RT between incongruent and congruent trials.
      final effect = (congRT != null && inconRT != null) ? inconRT - congRT : null;

      // Calculate total test duration.
      int? durationSeconds;
      if (_mainTestStartTime != null && _mainTestEndTime != null) {
        durationSeconds = _mainTestEndTime!.difference(_mainTestStartTime!).inSeconds;
      }

      // --- Construct the data object for saving ---
      final sessionId = "${DateTime.now().toIso8601String().replaceAll(':', '-').replaceAll('.', '-')}-$_participantId";
      final data = {
        // Session metadata for Joint Bayesian
        "session_id": sessionId,
        "app_version": "1.0.0",
        "device": "web",
        "start_time": _mainTestStartTime?.toIso8601String(),
        "end_time": _mainTestEndTime?.toIso8601String(),
        "duration_seconds": durationSeconds,
        
        // Compatibility fields
        "startTime": _mainTestStartTime?.toIso8601String(),
        "endTime": _mainTestEndTime?.toIso8601String(),
        "durationSeconds": durationSeconds,
        
        "trialData": _trialLog, // Raw trial-by-trial data.
        "resultsSummary": {
            "total":            _trialLog.length,
            "accuracy":         _acc(_trialLog),
            "mrt_total":        totalMRT?.round(),
            "mrt_cong":         congRT?.round(),
            "mrt_incong":       inconRT?.round(),
            "stroop_effect":    effect?.round(),
        },
        "config": {
          "reverse_button": _reverseBtn // Record the counterbalancing condition.
        },
        "submittedAt": FieldValue.serverTimestamp(),
      };

      // --- Save to Firestore ---
      final docRef = FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('cognitive_tests')
          .doc('stroop');

      await docRef.set(data, SetOptions(merge: true));

      if (!mounted) return;
      Navigator.of(context, rootNavigator: true).pop(); // Close the "Saving..." dialog.
      await Future.delayed(const Duration(milliseconds: 100)); // Ensure UI has time to update.
      if(mounted) {
        print("Stroop test completed for $_participantId. Returning to sequencer.");
        Navigator.pop(context); // Return to the previous page.
      }
    } catch (e) {
      if(mounted) {
        Navigator.pop(context); // Close dialog on error.
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('저장 실패: $e')), // "Save failed: ..."
        );
      }
      rethrow; // Rethrow for debugging.
    }
  }

  // -------------------------------------------------------------------------
  // Section 7: UI Building
  // -------------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    Widget? bodyContent;
    switch (_stage) {
      case StroopStage.welcome:
        // "Welcome", "This is the Stroop Task."
        bodyContent = _info("환영합니다", "이 검사는 Stroop 과제입니다.");
        break;
      case StroopStage.instructions:
        // "Task Instructions", "Please ignore the word's meaning and press the button for the ink color."
        bodyContent = _info("과제 안내", "단어 의미는 무시하고 글자색 버튼을 눌러주세요.");
        break;
      case StroopStage.practiceInstructions:
        // "Practice Instructions", "You will be presented with X practice trials."
        bodyContent = _info("연습 안내", "연습 과제 ${_practiceTrials.length}번이 제시됩니다.");
        break;
      case StroopStage.practiceTrial:
        final tr = _practiceTrials[_pIdx];
        bodyContent = _trial(tr["text"], tr["letterColor"], _onPracticeSelect, true);
        break;
      case StroopStage.practiceFeedback:
        bodyContent = _practiceFb();
        break;
      case StroopStage.mainTrialInstructions:
        // "Main Experiment Instructions", "There are a total of X trials.\nThere is a 3-second response limit for each trial."
        bodyContent = _info("본 시험 안내",
            "총 ${_trials.length} trial입니다.\n"
            "각 trial별 3초 응답 제한이 있습니다.");
        break;
      case StroopStage.mainTrial:
        if (_showFixation) {
          bodyContent = const Center(child: Text("+", style: TextStyle(fontSize: 48)));
        } else {
          if (_tIdx < _trials.length) {
             bodyContent = _buildMainTrialStimulus();
          } else {
             bodyContent = const Center(child: Text("로딩 중...")); // "Loading..."
          }
        }
        break;
      case StroopStage.result:
        bodyContent = _buildResultView();
        break;
    }

    return WillPopScope(
      onWillPop: () async => false, // Block back navigation.
      child: Scaffold(
        appBar: AppBar(
          title: const Text("Stroop 과제"), // "Stroop Task"
          backgroundColor: Colors.white,
        ),
        backgroundColor: Colors.white,
        body: Column(
          children: [
            // Show a progress bar during practice and main trials.
            if (_progressVal != null) LinearProgressIndicator(value: _progressVal),
            Expanded(
               child: Center( 
                 child: bodyContent ?? const Center(child: Text("Loading...")),
               ),
            ),
          ],
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Section 8: UI Helper Widgets
  // -------------------------------------------------------------------------

  /// Calculates the progress value for the linear progress indicator.
  double? get _progressVal {
    if (_stage == StroopStage.practiceTrial) return _pIdx / _practiceTrials.length;
    if (_stage == StroopStage.mainTrial)     return _tIdx / _trials.length;
    return null; // No progress bar on other screens.
  }

  /// A reusable widget for displaying information screens (welcome, instructions).
  Widget _info(String title, String body) {
    VoidCallback? onPressedAction = _nextStroopStage;
    String buttonText = "다음"; // "Next"

    // If on the welcome stage and there's an ID error, disable the 'Next' button.
    if (_stage == StroopStage.welcome && _hasIdError) {
       onPressedAction = null;
       buttonText = "오류: 진행 불가"; // "Error: Cannot proceed"
    }
    
    return SingleChildScrollView(
      key: ValueKey(title), // Key for AnimatedSwitcher
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(title, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Text(body, textAlign: TextAlign.center, style: const TextStyle(fontSize: 16)),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: onPressedAction,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[200],
              foregroundColor: Colors.black,
            ),
            child: Text(buttonText),
          ),
        ],
      ),
    );
  }

  /// Displays feedback after a practice trial.
  Widget _practiceFb() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // "Correct!" or "Incorrect!"
        Text(_pCorrect ? "맞았습니다!" : "틀렸습니다!", style: const TextStyle(fontSize: 32)),
        const SizedBox(height: 24),
        ElevatedButton(
          onPressed: _nextStroopStage,
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.grey[200],
            foregroundColor: Colors.black,
          ),
          child: const Text("다음"),
        ), // "Next"
      ],
    );
  }

  /// The main UI for a single trial, showing the stimulus and response buttons.
  Widget _trial(String txt, String colorKey,
      Function(String) onSel, bool enable) {
    final keys = _reverseBtn ? _colors.reversed : _colors;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          Text(txt, style: TextStyle(fontSize: 48, color: _toColor(colorKey))),
          const SizedBox(height: 30),
          if (enable)
            Wrap(
              spacing: 10, runSpacing: 10, alignment: WrapAlignment.center,
              children: keys.map((k) {
                return ElevatedButton(
                  onPressed: () => onSel(k),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[200],
                    foregroundColor: Colors.black,
                  ),
                  child: Text(_labelMap[k]!),
                );
              }).toList(),
            ),
        ],
      ),
    );
  }

  /// Builds the stimulus and response buttons for a main trial with precise timing.
  Widget _buildMainTrialStimulus() {
    final keys = _reverseBtn ? _colors.reversed : _colors;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          Text(_curTxt, style: TextStyle(fontSize: 48, color: _toColor(_curColorKey))),
          const SizedBox(height: 30),
          if (_canRespond)
            Wrap(
              spacing: 10, runSpacing: 10, alignment: WrapAlignment.center,
              children: keys.map((k) {
                return Listener(
                  onPointerDown: (event) => _onMainSelect(k, event),
                  child: ElevatedButton(
                    // onPressed must not be null for the button to be enabled,
                    // but the logic is handled by onPointerDown.
                    onPressed: () {},
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey[200],
                      foregroundColor: Colors.black,
                    ),
                    child: Text(_labelMap[k]!),
                  ),
                );
              }).toList(),
            ),
        ],
      ),
    );
  }

  /// The final screen shown at the end of the experiment.
  Widget _buildResultView() {
    return Center(
      key: const ValueKey('resultView'), // Key for AnimatedSwitcher.
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text( 
              "고생하셨습니다.", // "You have completed the task."
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: _saveResultsAndNavigate, 
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.grey[200],
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                textStyle: const TextStyle(fontSize: 16),
              ),
              child: const Text("저장 후 종료"), // "Save and Exit"
            ),
          ],
        ),
      ),
    );
  }
  
  // -------------------------------------------------------------------------
  // Section 9: Helper Methods
  // -------------------------------------------------------------------------
  /// Converts a color key string (e.g., "red") to a Material Color object.
  Color _toColor(String c) {
    switch (c) {
      case "red":    return Colors.red;
      case "green":  return Colors.green;
      case "blue":   return Colors.blue;
      case "yellow": return Colors.yellow;
      default:       return Colors.black;
    }
  }

  /// Calculates accuracy percentage.
  double _acc(List<Map<String, dynamic>> data) {
    if (data.isEmpty) return 0.0;
    final correct = data.where((t) => t["correct"] == true).length;
    return (correct / data.length) * 100.0;
  }

  /// Calculates mean reaction time for correct trials only using rt_ms field.
  double? _mrtMs(List<Map<String, dynamic>> data) {
    final correctTrials = data.where((t) => t["correct"] == true && t["rt_ms"] != null && (t["rt_ms"] as num) > 0);
    if (correctTrials.isEmpty) return null;
    final rts = correctTrials.map<double>((t) => (t["rt_ms"] as num).toDouble());
    return rts.reduce((a, b) => a + b) / rts.length;
  }

  /// Advances the experiment to the next stage.
  void _nextStroopStage() {
    // Prevent proceeding if the participant ID is invalid.
    if (_stage == StroopStage.welcome && _hasIdError) {
       // "Error: Cannot proceed because User ID is invalid."
       _showErrorAndPreventProceeding('오류: 사용자 ID가 유효하지 않아 진행할 수 없습니다.');
       return; 
    }
    
    // Prevent starting main trials if the trial list failed to generate.
    if (_stage == StroopStage.mainTrialInstructions && _trials.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        // "Error: Main experiment data was not generated."
        const SnackBar(content: Text('오류: 본 시험 데이터가 생성되지 않았습니다.')),
      );
      return;
    }

    setState(() {
      switch (_stage) {
        case StroopStage.welcome:               _stage = StroopStage.instructions;           break;
        case StroopStage.instructions:          _stage = StroopStage.practiceInstructions;   break;
        case StroopStage.practiceInstructions:  _stage = StroopStage.practiceTrial;          break;
        case StroopStage.practiceTrial:         break; // Handled by response selection.
        case StroopStage.practiceFeedback:
          if (_pIdx < _practiceTrials.length - 1) {
            _pIdx++; _stage = StroopStage.practiceTrial;
          } else {
            // Finished practice, move to main experiment.
            _stage = StroopStage.mainTrialInstructions;
          }
          break;
        case StroopStage.mainTrialInstructions: 
          _stage = StroopStage.mainTrial; 
          _startNextTrial(); 
          break;
        case StroopStage.mainTrial:             break; // StroopStage change is handled by _startNextTrial.
        case StroopStage.result:                break;
      }
    });
  }

  // -------------------------------------------------------------------------
  // Section 10: Trial Generation
  // -------------------------------------------------------------------------
  /// Builds the list of practice trials.
  /// [rand]: A seeded Random instance for reproducible randomization.
  List<Map<String, dynamic>> _buildPracticeTrials(Random rand) {
    // 3 congruent + 7 incongruent (10 total)
    final List<Map<String, dynamic>> list = [];
    // Congruent trials (3)
    for (var i = 0; i < 3; i++) {
      list.add({"text": _words[i], "letterColor": _colors[i], "type": "congruent"});
    }
    // Incongruent trials (7)
    while (list.length < 10) {
      final w = _words[rand.nextInt(4)];
      final c = _colors[rand.nextInt(4)];
      if (c == _word2color(w)) continue;  // Skip if congruent.
      list.add({"text": w, "letterColor": c, "type": "incongruent"});
    }
    return list;
  }

  /// Builds the list of main experiment trials (108 total).
  static List<Map<String, dynamic>> _buildMainTrials() {
    // 36 congruent, 36 neutral, 36 incongruent
    const int nCongruent   = 36;
    const int nNeutral     = 36;
    const int nIncongruent = 36;

    final List<Map<String, dynamic>> list = [];

    // 1. Congruent Trials: 4 combinations * 9 repetitions = 36
    final congruentSet = <Map<String, dynamic>>[];
    for (var j = 0; j < 4; j++) {
      congruentSet.add({"text": _words[j], "letterColor": _colors[j], "type": "congruent"});
    }
    for (var i = 0; i < nCongruent ~/ congruentSet.length; i++) {
      list.addAll(congruentSet);
    }

    // 2. Neutral Trials: 12 combinations * 3 repetitions = 36
    final neutralSet = <Map<String, dynamic>>[];
    for (var nw in _neutralWords) {
      for (var c in _colors) {
        neutralSet.add({"text": nw, "letterColor": c, "type": "neutral"});
      }
    }
    for (var i = 0; i < nNeutral ~/ neutralSet.length; i++) {
      list.addAll(neutralSet);
    }

    // 3. Incongruent Trials: 12 combinations * 3 repetitions = 36
    final incongruentSet = <Map<String, dynamic>>[];
    for (var w in _words) {
      for (var c in _colors) {
        if (c == _word2color(w)) continue; // Skip congruent pairs.
        incongruentSet.add({"text": w, "letterColor": c, "type": "incongruent"});
      }
    }
    for (var i = 0; i < nIncongruent ~/ incongruentSet.length; i++) {
      list.addAll(incongruentSet);
    }

    return list; // Total 108 trials.
  }

  /// Helper to map a Korean word string to an English color key.
  static String _word2color(String w) {
    switch (w) {
      case "빨강": return "red";
      case "초록": return "green";
      case "파랑": return "blue";
      case "노랑": return "yellow";
      default:     return "red"; // Should not happen.
    }
  }

  /// Shuffles a list of trials, ensuring no two consecutive trials are identical.
  /// [arr]: The list of trials to shuffle.
  /// [rand]: A seeded Random instance for reproducible shuffling.
  void _shuffleNoRepeat(List<Map<String, dynamic>> arr, [Random? rand]) {
    final random = rand ?? Random();
    // Helper to check for immediate repeats.
    bool hasRepeat() {
      for (var i = 1; i < arr.length; i++) {
        if (arr[i]["text"] == arr[i - 1]["text"] &&
            arr[i]["letterColor"] == arr[i - 1]["letterColor"]) return true;
      }
      return false;
    }
    // Keep shuffling until no immediate repeats are found.
    do { arr.shuffle(random); } while (hasRepeat());
  }
}
