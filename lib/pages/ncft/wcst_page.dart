// This file implements the Wisconsin Card Sorting Test (WCST).
// The WCST is a neuropsychological test of "set-shifting", i.e., the ability
// to display flexibility in the face of changing schedules of reinforcement.
//
// The task structure is as follows:
// 1. Four reference cards are displayed at the top of the screen.
// 2. The participant is presented with a series of stimulus cards and must
//    match each one to one of the reference cards.
// 3. The matching rule (color, shape, or number) is not explicitly told to
//    the participant. They must deduce it from "Correct" or "Incorrect" feedback.
// 4. After 10 consecutive correct responses, the sorting rule changes.
// 5. The test ends after 6 categories are completed or all 128 cards are used.
// 6. Various metrics are calculated, including total errors, perseverative errors,
//    and conceptual level responses.

// ---------------------------------------------------------------------------
// Section 0: External Packages and Imports
// ---------------------------------------------------------------------------
import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'dart:convert';
import 'package:crypto/crypto.dart' as crypto;
import 'package:flutter_svg/flutter_svg.dart';
import 'dart:html' as html;

// ---------------------------------------------------------------------------
// Section 1: WcstStage Definition and Constants
// ---------------------------------------------------------------------------
/// Defines the different stages of the WCST experiment.
enum WcstStage {
  welcome,              // Initial welcome screen.
  instructions,         // Task instructions screen.
  blockIntro1,          // Introduction before the first block of trials.
  blockTrials,          // Active trial presentation and response stage.
  blockTrialFeedback,   // Feedback screen shown after a trial.
  result,               // Final screen after the test is complete.
}

/// A top-level constant defining the fixed attributes of the four reference cards.
const List<Map<String, dynamic>> referenceAttrs = [
  {'name':'one_yellow_circle',   'count':1, 'color':'yellow', 'shape':'circle'},
  {'name':'two_black_rectangle', 'count':2, 'color':'black',  'shape':'rectangle'},
  {'name':'three_blue_star',     'count':3, 'color':'blue',   'shape':'star'},
  {'name':'four_red_triangle',   'count':4, 'color':'red',    'shape':'triangle'},
];

// ---------------------------------------------------------------------------
// Section 2: Root Widget
// ---------------------------------------------------------------------------
/// The main widget for the WCST screen.
class WcstPage extends StatefulWidget {
  const WcstPage({Key? key}) : super(key: key);

  @override
  State<WcstPage> createState() => _WcstPageState();
}

// ---------------------------------------------------------------------------
// Section 3: Helper Functions for Data Generation
// ---------------------------------------------------------------------------
/// Generates a deterministic 32-bit integer seed from a string participant ID.
/// This ensures that the card shuffling is the same for a given participant
/// if the test is ever retaken, while being different for other participants.
int _seedFromId(String id) {
  final bytes = utf8.encode(id);
  final hash = crypto.sha1.convert(bytes).bytes;
  // Use the first 4 bytes of the SHA1 hash to create a 32-bit integer.
  int seed = 0;
  for (var i = 0; i < 4; i++) {
    seed = (seed << 8) | hash[i];
  }
  return seed & 0x7fffffff; // Ensure the seed is a positive integer.
}

/// Generates the full 128-card deck for the WCST.
/// It creates one set of 64 unique cards, shuffles it based on the participant's
/// ID seed, and then duplicates it to create the 128-card deck.
List<Map<String, dynamic>> generateCardsData(String participantId) {
  const counts  = [1, 2, 3, 4];
  const colors  = ['yellow', 'black', 'blue', 'red'];
  const shapes  = ['circle', 'rectangle', 'star', 'triangle'];
  final List<Map<String, dynamic>> deck64 = [];

  for (final n in counts) {
    for (final c in colors) {
      for (final s in shapes) {
        final file = 'assets/images/$n$c$s.svg';
        deck64.add({
          'card'  : file,
          'count' : n, 
          'color': c, 
          'shape': s,
        });
      }
    }
  }
  deck64.shuffle(Random(_seedFromId(participantId)));
  final deck128 = [...deck64, ...deck64];
  return deck128;
}

// ---------------------------------------------------------------------------
// Section 4: State Class
// ---------------------------------------------------------------------------
/// Holds the state and business logic for the entire WCST experiment.
class _WcstPageState extends State<WcstPage> {
  // -------------------------------------------------------------------------
  // 4.1. Core State and Participant Variables
  // -------------------------------------------------------------------------
  String? _participantId;
  bool _isLoading = true; // Flag for initial data loading.
  WcstStage _stage = WcstStage.welcome;
  bool _handlingChoice = false; // Prevent double-tap race condition

  // -------------------------------------------------------------------------
  // 4.2. Card and Deck Management
  // -------------------------------------------------------------------------
  final List<String> choiceLabels = [
    "1번 카드", // "Card 1"
    "2번 카드", // "Card 2"
    "3번 카드", // "Card 3"
    "4번 카드", // "Card 4"
  ];
  final List<String> internalNames = [
    "one_yellow_circle",
    "two_black_rectangle",
    "three_blue_star",
    "four_red_triangle",
  ];

  List<Map<String, dynamic>>? _cardsData; // The full 128-card deck.
  late List<Map<String, dynamic>> _currentBlockData; // The current 64-card block.

  final List<Map<String, String>> _blockInfo = [
    {"blockName": "Deck1", "useRows": "0:64"},
    {"blockName": "Deck2", "useRows": "64:128"},
  ];
  int _currentBlockIndex = 0;
  int _currentTrialIndex = 0; // Index within the current 64-card block.
  int _usedCardCount = 0;     // Total cards used from the 128-card deck.

  // -------------------------------------------------------------------------
  // 4.3. Rule and Feedback Management
  // -------------------------------------------------------------------------
  final List<String> ruleCycle = ['colour', 'shape', 'number'];
  late String _currentRule;
  String _feedbackText = "";
  Color _feedbackColor = Colors.black;
  bool _lastChoiceCorrect = false;

  // -------------------------------------------------------------------------
  // 4.4. Performance Metrics and Scoring
  // -------------------------------------------------------------------------
  double? _trialStartTime; // High-precision timestamp for trial onset.
  DateTime? _testStartTime;
  DateTime? _testEndTime;
  
  // Core WCST metrics
  int _totalTrialCount = 0;
  int _totalCorrectCount = 0;
  int _perseverativeErrorCount = 0;
  int _nonPerseverativeErrorCount = 0;
  int _completedCategories = 0;
  int _failureToMaintainSet = 0;
  int _perseverativeResponseCount = 0;
  int _conceptualRespCount = 0;
  
  // Helper variables for metric calculation
  int _consecutiveCorrect = 0;
  bool _inFmsEligibleState = false; // True when 5+ consecutive correct but category not yet complete
  bool _fmsEpisodeActive = false; // Current FMS error episode in progress
  int? _trialsToCompleteFirstCategory;
  List<int> _trialsNeededForCategory = [];
  final List<String> _previousRules = [];
  int _trialsAttemptedForCurrentCategory = 0;
  int? _trialsToFirstConceptualResp;
  
  // Category-level CLR tracking for Heaton LTL
  List<int> _catTrials = [];
  List<int> _catConceptualCount = [];
  List<double> _catClrPercent = [];
  int _conceptualCountInThisCategory = 0;

  /// The final log of all trials to be saved.
  final List<Map<String, dynamic>> _trialLog = [];

  // -------------------------------------------------------------------------
  // Section 5: Widget Lifecycle and Initialization
  // -------------------------------------------------------------------------
  @override
  void initState() {
    super.initState();
    // Data initialization is moved to didChangeDependencies to access context.
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_participantId == null) { 
      // Initialize only once.
      final arg = ModalRoute.of(context)!.settings.arguments;
      if (arg is String && arg.isNotEmpty) {
        _participantId = arg;
      } else {
        // Fallback for testing or if ID is not passed correctly.
        _participantId = "unknown_pid_${Random().nextInt(100000)}";
        print("WCST_PAGE: Participant ID not available or not a valid string. Using fallback: $_participantId");
      }

      if (_participantId != null) {
        _cardsData = generateCardsData(_participantId!);
        _currentRule = ruleCycle[_completedCategories % ruleCycle.length];
        _setupBlockData(_currentBlockIndex);
        setState(() {
          _isLoading = false;
        });
      } else {
        print("WCST_PAGE: CRITICAL ERROR - Participant ID is null after attempting to set it.");
        setState(() {
          _isLoading = false; 
        });
      }
    }
  }

  @override
  void dispose() {
    // Clean up any resources if needed.
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Section 6: WcstStage and Trial Flow Logic
  // -------------------------------------------------------------------------
  
  /// Advances the experiment to the next logical stage.
  void _goNextWcstStage() {
    // For blockIntro1, we must call addPostFrameCallback *after* setState.
    if (_stage == WcstStage.blockIntro1) {
      setState(() {
        _testStartTime = DateTime.now(); // Record overall test start time.
        _stage = WcstStage.blockTrials;
        _trialsAttemptedForCurrentCategory = 0;
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _trialStartTime = html.window.performance.now();
      });
      return; // Exit to avoid the general setState below.
    }

    setState(() {
      switch (_stage) {
        case WcstStage.welcome:
          _stage = WcstStage.instructions;
          break;
        case WcstStage.instructions:
          _stage = WcstStage.blockIntro1;
          break;
        case WcstStage.blockIntro1:
          // This case is handled above and this path is not taken.
          break;
        case WcstStage.blockTrials:
          // This transition is now handled within _onChoiceSelected.
          // Kept here for logical flow, but effectively a no-op from button presses.
          _stage = WcstStage.blockTrialFeedback;
          break;
        case WcstStage.blockTrialFeedback:
          // After feedback, a new trial logic is initiated.
          _handleBlockTrialFeedback();
          break;
        case WcstStage.result:
          // No action needed here; flow stops at the result screen.
          break;
        default:
          break;
      }
    });
  }

  /// Automatically proceeds to the next stage after a 1-second delay.
  /// Used for showing feedback for a fixed duration.
  void _autoNextAfterDelay() {
    Future.delayed(const Duration(seconds: 1), _goNextWcstStage);
  }
  
  /// Sets up the data for the current block (deck of 64 cards).
  void _setupBlockData(int blockIndex) {
    if (_cardsData == null || blockIndex < 0 || blockIndex >= _blockInfo.length) {
      print("Error: _cardsData is null or blockIndex is out of bounds in _setupBlockData.");
      _currentBlockData = [];
      return;
    }
    final range = _blockInfo[blockIndex]["useRows"]!;
    final parts = range.split(":");
    final startIdx = int.parse(parts[0]);
    final endIdx = int.parse(parts[1]);
    
    _currentBlockData = _cardsData!.sublist(startIdx, endIdx); 
    _currentTrialIndex = 0; // Reset index for the new deck.
  }

  /// Logic to run after feedback is shown, preparing for the next trial.
  void _handleBlockTrialFeedback() {
    _trialsAttemptedForCurrentCategory++;

    if (_lastChoiceCorrect) {
      _consecutiveCorrect++;
      // Check for Conceptual Level Response (CLR)
      if (_consecutiveCorrect == 3) {
        _conceptualRespCount += 3;
        _conceptualCountInThisCategory += 3;
        if (_trialsToFirstConceptualResp == null) {
          _trialsToFirstConceptualResp = _totalTrialCount;
        }
      } else if (_consecutiveCorrect > 3) {
        _conceptualRespCount++;
        _conceptualCountInThisCategory++;
      }
      
      // Check for establishing a set (5 consecutive correct)
      if (_consecutiveCorrect == 5) {
        _inFmsEligibleState = true;
        _fmsEpisodeActive = false; // Ready for new FMS episode
      }
      
      // End FMS episode on correct response
      if (_fmsEpisodeActive) {
        _fmsEpisodeActive = false;
      }
    } else { // An error occurred
      _consecutiveCorrect = 0; // Reset correct streak.
      _inFmsEligibleState = false; // Reset FMS eligibility - must achieve 5 consecutive correct again
    }

    _currentTrialIndex++;
    _usedCardCount++;

    // Check for category completion (priority).
    if (_consecutiveCorrect >= 10) {
      _onCategoryFinished();
      return; // Stop further execution as _onCategoryFinished handles the next state.
    }

    // Check for end of test (all 128 cards used).
    if (_usedCardCount >= 128) {
      setState(() => _stage = WcstStage.result);
      return;
    }

    // Check for end of current deck.
    if (_currentTrialIndex >= _currentBlockData.length) {
      _gotoNextBlock(); 
    } else {
      // Proceed to the next trial in the same block.
      setState(() {
        _stage = WcstStage.blockTrials;
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _trialStartTime = html.window.performance.now();
      });
    }
  }

  /// Handles the logic when a category (10 consecutive correct) is completed.
  void _onCategoryFinished() {
    final String completedRule = _currentRule;
    _completedCategories++;
    _trialsNeededForCategory.add(_trialsAttemptedForCurrentCategory);
    _trialsToCompleteFirstCategory ??= _totalTrialCount;
    
    // Save category-level CLR statistics
    _catTrials.add(_trialsAttemptedForCurrentCategory);
    _catConceptualCount.add(_conceptualCountInThisCategory);
    final clrPct = (_trialsAttemptedForCurrentCategory > 0)
        ? (_conceptualCountInThisCategory / _trialsAttemptedForCurrentCategory) * 100.0
        : 0.0;
    _catClrPercent.add(clrPct);
    
    if (_previousRules.isEmpty || _previousRules.last != completedRule) {
      _previousRules.add(completedRule);
    }

    // Check for end of test (6 categories completed).
    if (_completedCategories >= 6) {
      setState(() => _stage = WcstStage.result);
      return;
    }

    // Update to the next rule in the cycle.
    _currentRule = ruleCycle[_completedCategories % ruleCycle.length];
    
    // Reset counters for the new category.
    _consecutiveCorrect = 0;
    _inFmsEligibleState = false;
    _fmsEpisodeActive = false;
    _trialsAttemptedForCurrentCategory = 0;
    _conceptualCountInThisCategory = 0;

    setState(() {
      _stage = WcstStage.blockTrials;
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _trialStartTime = html.window.performance.now();
    });
  }

  /// Moves to the next block of 64 cards.
  void _gotoNextBlock() {
    _currentBlockIndex++;
    if (_currentBlockIndex >= _blockInfo.length) {
      // This should not be reached if 128-card limit is hit first, but is a safeguard.
      setState(() => _stage = WcstStage.result);
    } else {
      _setupBlockData(_currentBlockIndex);
      setState(() {
        _stage = WcstStage.blockTrials;
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _trialStartTime = html.window.performance.now();
      });
    }
  }
  
  // -------------------------------------------------------------------------
  // Section 7: Response Evaluation and Data Logging
  // -------------------------------------------------------------------------

  /// Checks if a stimulus card matches a reference card on a given dimension.
  bool _matchOnDim(String dim, Map<String, dynamic> stim, Map<String, dynamic> ref) {
    switch (dim) {
      case 'colour':  return stim['color']  == ref['color'];
      case 'shape':   return stim['shape']  == ref['shape'];
      case 'number':  return stim['count']  == ref['count'];
    }
    return false;
  }

  /// Determines if a response qualifies as a perseverative response (using sandwich rule).
  bool _isPerseverativeResponse({
    required Map<String, dynamic> stim,
    required int chosenIdx,
    required String currRule,
    required String prevRule,
  }) {
    // A PR can only occur if there was a previous rule that is different from the current one.
    if (prevRule.isEmpty || currRule == prevRule) return false;

    final Map<String, dynamic> refChosen = referenceAttrs[chosenIdx];

    // Condition: The choice must match the stimulus on the *previous* rule's dimension.
    if (!_matchOnDim(prevRule, stim, refChosen)) return false;

    // "Sandwich Rule": The stimulus must match on the previous rule's dimension with *only one* reference card.
    int matchCount = 0;
    for (final refCardAttr in referenceAttrs) {
      if (_matchOnDim(prevRule, stim, refCardAttr)) {
        matchCount++;
      }
    }
    if (matchCount != 1) return false;

    return true; // All PR conditions are met.
  }

  /// Determines if an incorrect response qualifies as a perseverative error.
  bool _isPerseverativeError({
    required Map<String, dynamic> stim,
    required int chosenIdx,
    required String currRule,
    required String prevRule,
  }) {
    // A PE can only occur if there was a previous rule that is different from the current one.
    if (prevRule.isEmpty || currRule == prevRule) return false;

    final Map<String, dynamic> refChosen = referenceAttrs[chosenIdx];

    // Condition: The choice must match the stimulus on the *previous* rule's dimension.
    if (!_matchOnDim(prevRule, stim, refChosen)) return false;

    // "Sandwich Rule": The stimulus must match on the previous rule's dimension with *only one* reference card.
    int matchCount = 0;
    for (final refCardAttr in referenceAttrs) {
      if (_matchOnDim(prevRule, stim, refCardAttr)) {
        matchCount++;
      }
    }
    if (matchCount != 1) return false;

    return true; // All PE conditions are met.
  }

  /// The main function called when a participant selects a reference card.
  /// It evaluates the choice, calculates metrics, logs data, and transitions the stage.
  Future<void> _onChoiceSelected(int chosenIndex, Map<String, dynamic> cardData) async {
    if (_handlingChoice || _trialStartTime == null || _stage != WcstStage.blockTrials) return;
    _handlingChoice = true;
    
    try {
      // 1. Evaluate correctness and get RT.
      final double rt = html.window.performance.now() - _trialStartTime!;
      final chosenRefCard = referenceAttrs[chosenIndex];
      final bool isCorrect = _matchOnDim(_currentRule, cardData, chosenRefCard);
      
      // 2. Increment total trial count.
      _totalTrialCount++;

      // 3. Check for a Perseverative Response (PR) with sandwich rule.
      final String prevRule = _previousRules.isNotEmpty ? _previousRules.last : '';
      final bool isPR = _isPerseverativeResponse(
        stim: cardData,
        chosenIdx: chosenIndex,
        currRule: _currentRule,
        prevRule: prevRule,
      );
      if (isPR) {
        _perseverativeResponseCount++;
      }

      // 4. Update error counters if the choice was incorrect.
      bool isPE = false;
      bool isNPE = false;
      if (!isCorrect) {
        if (isPR) {
          // Perseverative Error: incorrect PR
          _perseverativeErrorCount++;
          isPE = true;
        } else {
          // Non-Perseverative Error
          _nonPerseverativeErrorCount++;
          isNPE = true;
          // FMS: Count only once per error episode when in eligible state
          if (_inFmsEligibleState && !_fmsEpisodeActive) {
            _failureToMaintainSet++;
            _fmsEpisodeActive = true; // Start FMS episode
          }
        }
      } else {
        _totalCorrectCount++;
      }
      _lastChoiceCorrect = isCorrect;

      // 5. Log all data for the current trial.
      _trialLog.add({
        // Common schema fields for Joint Bayesian
        'participant_id': _participantId,
        'task': 'wcst',
        'trial_index': _totalTrialCount - 1,
        'block_index': _currentBlockIndex,
        'stim_onset_ms': _trialStartTime,  // Trial onset time
        'resp_time_ms': _trialStartTime != null ? _trialStartTime! + rt : null,
        'rt_ms': rt,
        'correct': _lastChoiceCorrect,
        'timeout': false,  // WCST doesn't have timeouts
        'cond': _currentRule,  // colour/shape/number
        
        // Extra task-specific fields
        'extra': {
          'trialIndexInBlock': _currentTrialIndex,
          'stageName': _stage.toString(),
          'chosenCard': internalNames[chosenIndex],
          'ruleAtThatTime': _currentRule,
          'cardNumber': cardData['count'],
          'cardColor': cardData['color'],
          'cardShape': cardData['shape'],
          'targetCardPath': cardData['card'],
          'isPR': isPR,
          'isPE': isPE,
          'isNPE': isNPE,
        },
        
        // Legacy compatibility fields
        'trialIndex': _totalTrialCount - 1,
        'trialIndexInBlock': _currentTrialIndex,
        'stageName': _stage.toString(),
        'chosenCard': internalNames[chosenIndex],
        'ruleAtThatTime': _currentRule,
        'reactionTimeMs': rt,
        'cardNumber': cardData['count'],
        'cardColor': cardData['color'],
        'cardShape': cardData['shape'],
        'targetCardPath': cardData['card'],
        'timestamp': DateTime.now().toIso8601String(),
        'isPR': isPR,
        'isPE': isPE,
        'isNPE': isNPE,
      });

      // 6. Set feedback and transition to the feedback stage.
      setState(() {
        _feedbackText = _lastChoiceCorrect ? "정답" : "오답"; // "Correct" : "Incorrect"
        _feedbackColor = Colors.black;
        _stage = WcstStage.blockTrialFeedback; // Change state immediately to prevent race condition
      });

      _autoNextAfterDelay();
    } finally {
      _handlingChoice = false;
    }
  }
  
  // -------------------------------------------------------------------------
  // Section 8: Data Saving and Finalization
  // -------------------------------------------------------------------------
  
  /// Saves the final results to Firestore and navigates away from the test.
  Future<void> _saveResultAndFinish() async {
    // Show a "Saving..." dialog.
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Dialog(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            CircularProgressIndicator(),
            SizedBox(width: 20),
            Text("결과 저장 중...") // "Saving results..."
          ]),
        ),
      ),
    );

    try {
      final resultSummary = _calculateFinalStats();
      _testEndTime = DateTime.now();
      int durationSeconds = 0;
      if (_testStartTime != null && _testEndTime != null) {
        durationSeconds = _testEndTime!.difference(_testStartTime!).inSeconds;
      }

      // Path: /participants/{participantId}/cognitive_tests/wcst
      final wcstDocRef = FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('cognitive_tests')
          .doc('wcst');

      final sessionId = "${DateTime.now().toIso8601String().replaceAll(':', '-').replaceAll('.', '-')}-$_participantId";
      final dataToSave = {
        // Session metadata for Joint Bayesian
        "session_id": sessionId,
        "app_version": "1.0.0",
        "device": "web",
        "start_time": _testStartTime?.toIso8601String(),
        "end_time": _testEndTime?.toIso8601String(),
        "duration_seconds": durationSeconds,
        
        // Legacy fields (kept for compatibility)
        "startTime": _testStartTime?.toIso8601String(),
        "endTime": _testEndTime?.toIso8601String(),
        "durationSeconds": durationSeconds,
        
        "resultsSummary": resultSummary,
        "trialData": _trialLog,
        "submittedAt": FieldValue.serverTimestamp(),
      };

      await wcstDocRef.set(dataToSave, SetOptions(merge: true));

      if (!mounted) return;
      Navigator.of(context, rootNavigator: true).pop(); // Close dialog.
      
      print("WCST test completed for $_participantId. Returning to sequencer.");
      Navigator.of(context).pop(); // Return to previous page.

    } catch (e, s) {
      if (mounted) {
        Navigator.of(context, rootNavigator: true).pop(); // Close dialog on error.
        ScaffoldMessenger.of(context).showSnackBar(
          // "Failed to save results. Please contact administrator. Error: ..."
          SnackBar(content: Text("결과 저장에 실패했습니다. 관리자에게 문의하세요. 오류: $e")),
        );
        print("WCST Save Error: $e");
        print("Stack trace: $s");
      }
    }
  }

  /// Calculates the final summary statistics for the results document.
  Map<String, dynamic> _calculateFinalStats() {
    final totalErrorCount = _perseverativeErrorCount + _nonPerseverativeErrorCount;
    
    // Trial-based efficiency metric (previous "Learning to learn")
    double learningEfficiencyDeltaTrials = 0.0;
    if (_trialsNeededForCategory.length >= 6) {
      final firstHalf = (_trialsNeededForCategory[0] +
          _trialsNeededForCategory[1] +
          _trialsNeededForCategory[2]) / 3.0;
      final secondHalf = (_trialsNeededForCategory[3] +
          _trialsNeededForCategory[4] +
          _trialsNeededForCategory[5]) / 3.0;
      learningEfficiencyDeltaTrials = firstHalf - secondHalf;
    }
    
    // Heaton-style Learning-to-Learn (CLR% based)
    double learningToLearnHeatonClrDelta = 0.0;
    if (_catClrPercent.length >= 6) {
      final f3_clr = (_catClrPercent[0] + _catClrPercent[1] + _catClrPercent[2]) / 3.0;
      final l3_clr = (_catClrPercent[3] + _catClrPercent[4] + _catClrPercent[5]) / 3.0;
      learningToLearnHeatonClrDelta = l3_clr - f3_clr; // Heaton: improvement = last3 - first3
    }
    
    // % Conceptual Level Responses (CLR)
    double clrPercent = 0.0;
    if (_totalTrialCount > 0) {
      clrPercent = (_conceptualRespCount / _totalTrialCount) * 100;
    }

    // % Perseverative Responses (PR)
    double prPercent = 0.0;
    if (_totalTrialCount > 0) {
      prPercent = (_perseverativeResponseCount / _totalTrialCount) * 100;
    }

    // Handle both 1-based and 0-based versions of trialsToFirstConceptualResp
    final hasFirstCLR = _trialsToFirstConceptualResp != null;
    final firstClr1Based = _trialsToFirstConceptualResp;  // 1-based (human-readable)
    final firstClr0Based = hasFirstCLR ? (_trialsToFirstConceptualResp! - 1) : -1;  // 0-based for log matching

    return {
      "totalTrialCount": _totalTrialCount,
      "totalCorrectCount": _totalCorrectCount,
      "totalErrorCount": totalErrorCount,
      "perseverativeErrorCount": _perseverativeErrorCount,
      "nonPerseverativeErrorCount": _nonPerseverativeErrorCount,
      "completedCategories": _completedCategories,
      "trialsToCompleteFirstCategory": _trialsToCompleteFirstCategory ?? 0,
      "failureToMaintainSet": _failureToMaintainSet,
      "learningEfficiencyDeltaTrials": learningEfficiencyDeltaTrials,
      "learningToLearnHeatonClrDelta": learningToLearnHeatonClrDelta,
      "categoryClrPercents": _catClrPercent,
      "conceptualLevelResponses": _conceptualRespCount,
      "hasFirstCLR": hasFirstCLR,
      "trialsToFirstConceptualResp": firstClr1Based ?? 0,  // 1-based (legacy compatibility)
      "trialsToFirstConceptualResp0": firstClr0Based,  // 0-based for log index matching
      "conceptualLevelResponsesPercent": clrPercent,
      "perseverativeResponses": _perseverativeResponseCount,
      "perseverativeResponsesPercent": prPercent,
      "timestamp": DateTime.now().toIso8601String(),
    };
  }

  // -------------------------------------------------------------------------
  // Section 9: UI Building
  // -------------------------------------------------------------------------
  
  /// Calculates the progress value for the linear progress indicator.
  double? _getProgressValue() {
    if (_stage == WcstStage.blockTrials || _stage == WcstStage.blockTrialFeedback) {
      const maxTrials = 128;
      final current = (_usedCardCount + 1).clamp(1, maxTrials);
      return current / maxTrials;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text("WCST")),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    if (_cardsData == null) {
      return Scaffold(
        appBar: AppBar(title: const Text("WCST")),
        body: const Center(
          child: Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              "Error: Could not load test data. Participant ID might be missing or invalid.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.red, fontSize: 16),
            ),
          ),
        ),
      );
    }

    return WillPopScope(
      onWillPop: () async => false, // Block back navigation.
      child: Scaffold(
        appBar: AppBar(
          title: const Text("WCST 과제"), // "WCST Task"
          backgroundColor: Colors.white,
        ),
        backgroundColor: Colors.white,
        body: Column(
          children: [
            if (_getProgressValue() != null)
              LinearProgressIndicator(value: _getProgressValue()),
            Expanded(child: _buildCurrentWcstStageUI()),
          ],
        ),
      ),
    );
  }

  /// The main UI router. Returns the widget corresponding to the current experiment stage.
  Widget _buildCurrentWcstStageUI() {
    switch (_stage) {
      case WcstStage.welcome:
        return _buildInfoScreen(
          title: "환영합니다", // "Welcome"
          content: "이 실험은 Wisconsin Card Sorting Task (WCST)입니다.", // "This is the Wisconsin Card Sorting Task (WCST)."
          onNext: _goNextWcstStage,
        );
      case WcstStage.instructions:
        return _buildInfoScreen(
          title: "WCST 안내", // "WCST Instructions"
          content:
          "카드를 색상, 모양, 숫자 규칙으로 분류합니다.\n" // "You will sort cards by color, shape, or number rules."
              "규칙은 사전에 알려주지 않으며, 맞으면 '정답', 틀리면 '오답'만 표시됩니다.\n\n" // "The rule is not told to you. You will only see 'Correct' or 'Incorrect'."
              "연속 10회 정답 시 규칙이 변경되고,\n" // "The rule changes after 10 consecutive correct answers."
              "최대 6개 카테고리를 완료하거나 128장을 모두 소진하면 종료됩니다.", // "The test ends after 6 categories or when all 128 cards are used."
          onNext: _goNextWcstStage,
        );
      case WcstStage.blockIntro1:
        return _buildInfoScreen(
          title: "본 실험 안내", // "Main Experiment Instructions"
          content: "이 실험은 예비실험이 없고 바로 본 실험으로 진행됩니다.\n하단 버튼을 눌러 진행하세요.",
          onNext: _goNextWcstStage,
        );

      case WcstStage.blockTrials:
        return _buildBlockTrialUI();

      case WcstStage.blockTrialFeedback:
        return _buildFeedbackUI();

      case WcstStage.result:
        return _buildResultScreen();
    }
  }

  /// A reusable widget for displaying information screens.
  Widget _buildInfoScreen({
    required String title,
    required String content,
    required VoidCallback onNext,
  }) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Text(title, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            Text(content, style: const TextStyle(fontSize: 16), textAlign: TextAlign.center),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: onNext,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.grey[200],
                foregroundColor: Colors.black,
              ),
              child: const Text("다음으로"), // "Next"
            ),
          ],
        ),
      ),
    );
  }

  /// Builds the UI for an active trial.
  Widget _buildBlockTrialUI() {
    if (_currentBlockData.isEmpty || _currentTrialIndex >= _currentBlockData.length) {
      return const Center(child: Text("Loading new block or trial finished..."));
    }
    final cardData = _currentBlockData[_currentTrialIndex];
    return _buildSingleTrialWidget(cardData, "Trial ${_usedCardCount + 1} / 128 (Deck ${_currentBlockIndex + 1})");
  }

  /// Builds the specific layout for a single trial with reference cards and a stimulus card.
  Widget _buildSingleTrialWidget(Map<String, dynamic> cardData, String title) {
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            _buildReferenceCards(),
            const SizedBox(height: 50),
            SizedBox(
              width: 200,
              height: 200,
              child: SvgPicture.asset(cardData['card'], fit: BoxFit.contain),
            ),
            const SizedBox(height: 30),
            const Text(
              "위 카드를 어느 기준 카드 아래로 분류하시겠습니까?", // "To which reference card would you classify the card above?"
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 30),
            Wrap(
              spacing: 4,
              runSpacing: 4,
              alignment: WrapAlignment.center,
              children: List.generate(choiceLabels.length, (idx) {
                return ElevatedButton(
                  onPressed: () => _onChoiceSelected(idx, cardData),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[200],
                    foregroundColor: Colors.black,
                  ),
                  child: Text(choiceLabels[idx]),
                );
              }),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds the row of four static reference cards.
  Widget _buildReferenceCards() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _buildCardImage("assets/images/1yellowcircle.svg"),
        _buildCardImage("assets/images/2blackrectangle.svg"),
        _buildCardImage("assets/images/3bluestar.svg"),
        _buildCardImage("assets/images/4redtriangle.svg"),
      ],
    );
  }

  /// Helper to build a single card image widget.
  Widget _buildCardImage(String path) {
    return Container(
      margin: const EdgeInsets.all(10),
      width: 120,
      height: 120,
      child: SvgPicture.asset(path, fit: BoxFit.contain),
    );
  }

  /// Builds the feedback UI ("Correct" / "Incorrect").
  Widget _buildFeedbackUI() {
    return Center(
      child: Text(
        _feedbackText,
        style: TextStyle(
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: _feedbackColor,
        ),
      ),
    );
  }

  /// Builds the final result screen.
  Widget _buildResultScreen() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            "고생하셨습니다.", // "You have completed the task."
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _saveResultAndFinish,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[200],
              foregroundColor: Colors.black,
            ),
            child: const Text("결과 저장 후 종료"), // "Save Results and Exit"
          )
        ],
      ),
    );
  }
}
