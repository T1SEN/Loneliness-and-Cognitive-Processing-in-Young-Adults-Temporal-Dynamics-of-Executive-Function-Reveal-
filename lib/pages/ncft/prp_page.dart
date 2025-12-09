// This file implements the Psychological Refractory Period (PRP) task.
// The PRP effect occurs when two stimuli are presented in close succession,
// and the response to the second stimulus is delayed. This task measures that delay.
//
// The task structure is as follows:
// 1. T1: A number is presented. Participant judges if it's odd or even.
// 2. T2: A colored square is presented after a variable Stimulus Onset Asynchrony (SOA).
//    Participant identifies the color.
// 3. The experiment includes practice trials, main trials, and rest breaks.
// 4. Data (responses, RTs, accuracy) is logged and saved to Firestore.

// ---------------------------------------------------------------------------
// Section 0: External packages
// ---------------------------------------------------------------------------
import 'dart:async';
import 'dart:html' as html; // Used for high-precision timing (window.performance.now()) on the web.
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

// ---------------------------------------------------------------------------
// Section 1: Root Widget
// ---------------------------------------------------------------------------
/// The main widget for the PRP task screen.
class PrpPage extends StatefulWidget {
  const PrpPage({super.key});
  @override
  State<PrpPage> createState() => _PrpPageState();
}

// ---------------------------------------------------------------------------
// Section 2: Experiment PrpStage Enum
// ---------------------------------------------------------------------------
/// Defines the different stages or phases of the experiment.
/// This controls which UI is displayed and what logic is executed.
enum PrpStage {
  welcome,              // Initial welcome screen.
  instructions,         // Screen with general instructions.
  practiceInstructions, // Screen with instructions for the practice block.
  practiceTrial,        // An active practice trial is in progress.
  practiceFeedback,     // Feedback shown after a practice trial.
  mainTrialInstructions,// Screen with instructions for the main experiment block.
  mainTrial,            // An active main trial is in progress.
  restBreak,            // A rest break period between blocks of main trials.
  result,               // Final screen shown after the experiment is complete.
}

// ---------------------------------------------------------------------------
// Section 3: State Class
// ---------------------------------------------------------------------------
/// Holds the state and business logic for the entire PRP experiment.
class _PrpPageState extends State<PrpPage> {
  // -------------------------------------------------------------------------
  // 3.1. Participant and Key Mapping Variables
  // -------------------------------------------------------------------------
  
  /// A unique identifier for the participant.
  late String _participantId;
  
  /// Flags for counterbalancing response key mappings to avoid stimulus-response compatibility effects.
  /// Based on the participant ID hash, keys for T1 and T2 are pseudo-randomly reversed.
  late final bool _reverseT1Keys;
  late final bool _reverseT2Keys;
  
  /// Participant-specific mapping of physical response locations (left/right button)
  /// to the logical response codes for Task 1 ('O' for Odd, 'E' for Even).
  late String _leftRespT1,  _rightRespT1;
  
  /// Participant-specific mapping for Task 2 ('R' for Red, 'B' for Blue).
  late String _leftRespT2,  _rightRespT2;

  /// Flag to ensure initialization logic runs only once.
  bool _initialized = false;

  // -------------------------------------------------------------------------
  // 3.2. Experiment PrpStage and Control
  // -------------------------------------------------------------------------

  /// The current stage of the experiment, from the `PrpStage` enum.
  PrpStage _stage = PrpStage.welcome;

  /// Random number generator for shuffling trials and selecting stimuli.
  final Random _rand = Random();

  // -------------------------------------------------------------------------
  // 3.3. Fixed Experiment Parameters
  // -------------------------------------------------------------------------

  /// List of Stimulus Onset Asynchronies (SOAs) in milliseconds.
  /// SOA is the time between the onset of T1 and the onset of T2.
  static const List<int> _soas = [50, 150, 300, 600, 1200];
  
  /// Duration of the fixation cross in milliseconds.
  static const int _fixDur = 500;
  
  /// Timeout for responding to Task 1 in milliseconds.
  static const int _t1TO   = 3000;
  
  /// Timeout for responding to Task 2 in milliseconds.
  static const int _t2TO   = 3000;
  
  /// Duration of feedback display after practice trials in milliseconds.
  static const int _fbDur  = 1000;
  
  /// Duration of the inter-trial interval (ITI) in milliseconds.
  static const int _itiDur = 500;
  
  /// Number of practice trials (5 SOAs * 2 repetitions).
  static const int _nPracticeTrials = 10;
  
  /// Total number of main trials (5 SOAs * 2 T2 colors * 2 T1 parities * 6 repetitions = 120).
  static const int _totalTrials = 120;

  // -------------------------------------------------------------------------
  // 3.4. Internal State Variables for Trial Management
  // -------------------------------------------------------------------------

  /// List of trial configurations for the practice block.
  late final List<Map<String, dynamic>> _practiceTrials;
  
  /// List of trial configurations for the main experiment.
  late final List<Map<String, dynamic>> _trials;
  
  /// Index of the current practice trial.
  int _pIdx = 0;
  
  /// Index of the current main trial.
  int _tIdx = 0;
  
  /// Data for providing feedback after a practice trial.
  Map<String, dynamic>? _pFeedbackData;

  // -------------------------------------------------------------------------
  // 3.5. Live State Variables for the Current Trial
  // -------------------------------------------------------------------------

  // Stimulus and response UI display flags
  bool _showFix = false;  // Show fixation cross?
  bool _showT1  = false;  // Show T1 stimulus?
  bool _showT2  = false;  // Show T2 stimulus?

  // Current trial's stimulus properties
  int? _currT1;           // T1 stimulus (a number from 1-9).
  String? _currT2;        // T2 stimulus ("red" or "blue").
  int _currSOA = 0;       // SOA for the current trial.

  // Response tracking
  bool _canRespT1 = false; // Is the participant allowed to respond to T1?
  bool _canRespT2 = false; // Is the participant allowed to respond to T2?
  String? _t1Resp;         // The participant's response for T1.
  String? _t2Resp;         // The participant's response for T2.
  double? _rt1;            // Reaction time for T1.
  double? _rt2;            // Reaction time for T2.
  bool _t1Done = false;    // Has T1 response been registered?
  bool _t2Done = false;    // Has T2 response been registered?
  
  // Precision timing variables
  double? _trialStartAbs;  // Trial start absolute time (ms, performance.now)
  double? _t1RespAbs;      // T1 response absolute time
  double? _t2RespAbs;      // T2 response absolute time
  bool _t2PressedWhileT1Pending = false;  // Was T2 pressed before T1 completed?

  // -------------------------------------------------------------------------
  // 3.6. Timers
  // -------------------------------------------------------------------------
  /// Timestamps for calculating reaction times. Uses high-precision timer.
  double? _t1StartTime, _t2StartTime;
  
  /// Timers to control the sequence of events within a trial.
  Timer? _fixTimer;   // Controls fixation duration.
  Timer? _soaTimer;   // Controls the SOA period before showing T2.
  Timer? _t1ToTimer;  // Timeout for T1 response.
  Timer? _t2ToTimer;  // Timeout for T2 response.
  Timer? _fbTimer;    // Controls feedback duration.
  Timer? _itiTimer;   // Controls inter-trial interval.

  // -------------------------------------------------------------------------
  // 3.7. Rest Break State
  // -------------------------------------------------------------------------
  Timer? _restTimer;        // Timer for the rest break countdown.
  int _restCountdown = 10;  // Countdown duration in seconds.

  // -------------------------------------------------------------------------
  // 3.8. Data Logging and Experiment Timing
  // -------------------------------------------------------------------------
  /// A list of maps, where each map stores the data for one main trial.
  final List<Map<String, dynamic>> _log = [];
  
  /// The time the first main trial started. Used to calculate total duration.
  DateTime? _experimentStartTime;

  // -------------------------------------------------------------------------
  // 3.9. Keyboard Input Management and Visual Feedback
  // -------------------------------------------------------------------------
  /// Focus node for capturing keyboard input
  final FocusNode _kbdFocus = FocusNode();
  
  /// Visual feedback states for key presses
  bool _aKeyPressed = false;
  bool _dKeyPressed = false;
  bool _leftKeyPressed = false;
  bool _rightKeyPressed = false;

  /// Track currently pressed keys to prevent auto-repeat responses
  final Set<LogicalKeyboardKey> _pressedKeys = {};

  // -------------------------------------------------------------------------
  // Section 3.9: Widget Lifecycle Methods (initState, dispose)
  // -------------------------------------------------------------------------
  @override
  void initState() {
    super.initState();
  }

  /// This method is called when the widget's dependencies change.
  /// It's used here for initialization that requires `context`, which is not
  /// available in `initState`.
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_initialized) return;

    // Retrieve participant ID from the arguments passed to this page.
    final arg = ModalRoute.of(context)!.settings.arguments;
    _participantId = arg is String ? arg : 'unknown';

    // Generate a hash from the participant ID to seed the counterbalancing.
    final int hash = _participantId.codeUnits.fold(0, (a, b) => a + b);
    
    // Assign key mappings based on the hash (50% chance for each reversal).
    _reverseT1Keys = (hash % 4) < 2;
    _reverseT2Keys = (hash % 2) == 1;

    // Determine the correct response codes for left/right keys for this participant.
    // T1 Task: 'O' = Odd, 'E' = Even
    _leftRespT1  = _reverseT1Keys ? 'E' : 'O';
    _rightRespT1 = _reverseT1Keys ? 'O' : 'E';
    // T2 Task: 'R' = Red, 'B' = Blue
    _leftRespT2  = _reverseT2Keys ? 'B' : 'R';
    _rightRespT2 = _reverseT2Keys ? 'R' : 'B';

    // Generate the trial lists.
    _practiceTrials = _buildPracticeTrials();
    _trials         = _buildMainTrials();
    _initialized = true;
  }

  @override
  void dispose() {
    // Cancel all active timers to prevent memory leaks and errors when the widget is removed.
    for (final t in [_fixTimer, _soaTimer, _t1ToTimer, _t2ToTimer, _fbTimer, _itiTimer, _restTimer]) { t?.cancel(); }
    _kbdFocus.dispose();  // Dispose of the focus node
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Section 3.10: Keyboard Input Handler
  // -------------------------------------------------------------------------
  /// Handles keyboard input for T1 (A/D keys) and T2 (arrow keys) responses
  void _onKey(RawKeyEvent e) {
    // Only process during active trial stages
    if (!(_stage == PrpStage.practiceTrial || _stage == PrpStage.mainTrial)) return;
    
    final key = e.logicalKey;
    
    if (e is RawKeyDownEvent) {
      // Ignore if this key is already pressed (prevents auto-repeat)
      if (_pressedKeys.contains(key)) return;
      
      // Mark this key as pressed
      _pressedKeys.add(key);
      
      // Handle key press visual feedback only when relevant stimuli are visible
      setState(() {
        if (key == LogicalKeyboardKey.keyA && _showT1) {
          _aKeyPressed = true;
        } else if (key == LogicalKeyboardKey.keyD && _showT1) {
          _dKeyPressed = true;
        } else if (key == LogicalKeyboardKey.arrowLeft && _showT2) {
          _leftKeyPressed = true;
        } else if (key == LogicalKeyboardKey.arrowRight && _showT2) {
          _rightKeyPressed = true;
        }
      });

      // T1 responses: A (left) / D (right) - Always enabled when T1 is shown
      if (_showT1 && !_t1Done) {
        if (key == LogicalKeyboardKey.keyA) {
          _handleResponse(isT1: true, response: _leftRespT1);
          return;
        } else if (key == LogicalKeyboardKey.keyD) {
          _handleResponse(isT1: true, response: _rightRespT1);
          return;
        }
      }

      // T2 responses: ← (left) / → (right) - Always enabled when T2 is shown
      if (_showT2 && !_t2Done) {
        if (key == LogicalKeyboardKey.arrowLeft) {
          _handleResponse(isT1: false, response: _leftRespT2);
          return;
        } else if (key == LogicalKeyboardKey.arrowRight) {
          _handleResponse(isT1: false, response: _rightRespT2);
          return;
        }
      }
    } else if (e is RawKeyUpEvent) {
      // Remove key from pressed set and reset visual feedback
      _pressedKeys.remove(key);
      
      setState(() {
        if (key == LogicalKeyboardKey.keyA) {
          _aKeyPressed = false;
        } else if (key == LogicalKeyboardKey.keyD) {
          _dKeyPressed = false;
        } else if (key == LogicalKeyboardKey.arrowLeft) {
          _leftKeyPressed = false;
        } else if (key == LogicalKeyboardKey.arrowRight) {
          _rightKeyPressed = false;
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // Section 4: Trial Generation
  // -------------------------------------------------------------------------
  
  /// Builds the list of main experiment trials.
  ///
  /// Creates a fully-crossed and balanced design:
  /// 6 repetitions × 5 SOAs × 2 T2 colors × 2 T1 parities = 120 trials.
  List<Map<String, dynamic>> _buildMainTrials() {
    final List<Map<String, dynamic>> trials = [];
    const oddNums  = [1, 3, 5, 7, 9];
    const evenNums = [2, 4, 6, 8];
    const t2Stim   = ['red', 'blue'];

    for (int rep = 0; rep < 6; rep++) {
      for (final soa in _soas) {
        for (final colour in t2Stim) {
          for (final parity in [true, false]) { // true: odd, false: even
            // Select a random number with the correct parity.
            final t1 = parity ? oddNums[_rand.nextInt(oddNums.length)]
                              : evenNums[_rand.nextInt(evenNums.length)];
            trials.add({
              't1_stim'   : t1,
              't1_correct': parity ? 'O' : 'E', // Correct response for T1
              't2_stim'   : colour,
              't2_correct': colour == 'red' ? 'R' : 'B', // Correct response for T2
              'soa'       : soa,
            });
          }
        }
      }
    }
    trials.shuffle(_rand); // Randomize the trial order.
    return trials;
  }

  /// Builds the list of practice trials.
  ///
  /// 2 repetitions × 5 SOAs = 10 trials. Stimuli are chosen randomly.
  List<Map<String, dynamic>> _buildPracticeTrials() {
    final List<Map<String, dynamic>> p = [];
    const oddNums  = [1, 3, 5, 7, 9];
    const evenNums = [2, 4, 6, 8];
    for (int rep = 0; rep < 2; rep++) {
      for (final soa in _soas) {
        final bool parity = _rand.nextBool();
        final int t1 = parity ? oddNums[_rand.nextInt(oddNums.length)]
                              : evenNums[_rand.nextInt(evenNums.length)];
        final colour = _rand.nextBool() ? 'red' : 'blue';
        p.add({
          't1_stim'   : t1,
          't1_correct': parity ? 'O' : 'E',
          't2_stim'   : colour,
          't2_correct': colour == 'red' ? 'R' : 'B',
          'soa'       : soa,
        });
      }
    }
    p.shuffle(_rand);
    return p;
  }

  // -------------------------------------------------------------------------
  // Section 5: Trial Flow and Execution Logic
  // -------------------------------------------------------------------------
  
  /// Initiates the next practice trial or moves to the main experiment if all
  /// practice trials are complete.
  void _startNextPracticeTrial() {
    if (!mounted) return;
    if (_pIdx >= _practiceTrials.length) {
      // Finished practice, move to main trial instructions.
      setState(() => _stage = PrpStage.mainTrialInstructions);
      return;
    }
    _runTrial(_practiceTrials[_pIdx]);
  }

  /// Initiates the next main trial, handles rest breaks, or moves to the
  /// results screen if all trials are complete.
  void _startNextMainTrial() {
    if (!mounted) return;
    if (_tIdx >= _trials.length) {
      // Finished all main trials, move to results screen.
      setState(() => _stage = PrpStage.result);
      return;
    }
    if (_tIdx == 0) { // Record start time at the beginning of the first main trial.
      _experimentStartTime = DateTime.now();
    }
    
    // Check if it's time for a rest break (after trials 30, 60, 90).
    if (_tIdx > 0 && _tIdx % 30 == 0 && _tIdx < _trials.length) {
      _startRestBreak();
      return;
    }
    
    // Run the next trial.
    _runTrial(_trials[_tIdx]);
  }

  /// Sets up and runs a single trial (both practice and main).
  ///
  /// [trial]: A map containing the configuration for the current trial.
  void _runTrial(Map<String, dynamic> trial) {
    // Capture trial start time immediately
    _trialStartAbs = html.window.performance.now();
    
    // Reset all live trial state variables.
    setState(() {
      _currT1 = trial['t1_stim'] as int;
      _currT2 = trial['t2_stim'] as String;
      _currSOA= trial['soa'] as int;
      _showFix = true;
      _showT1  = _showT2 = false;
      _canRespT1 = _canRespT2 = false;
      _t1Resp = _t2Resp = null;
      _rt1 = _rt2 = null;
      _t1Done = _t2Done = false;
      _pFeedbackData = null;
      _t1StartTime = null;
      _t2StartTime = null;
      _t1RespAbs = _t2RespAbs = null;
      _t2PressedWhileT1Pending = false;
      
      // Reset visual feedback states
      _aKeyPressed = false;
      _dKeyPressed = false;
      _leftKeyPressed = false;
      _rightKeyPressed = false;
    });

    // Clear pressed keys set
    _pressedKeys.clear();

    // Request keyboard focus after the frame is drawn
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _kbdFocus.requestFocus();
    });

    // Start the trial sequence with a fixation cross.
    _fixTimer = Timer(Duration(milliseconds: _fixDur), () {
      if (!mounted) return;
      // After fixation, show T1 (responses are always enabled when shown)
      setState(() { _showFix = false; _showT1 = true; _canRespT1 = true; });
      
      // Schedule event timing logic for after the frame is rendered.
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        _t1StartTime = html.window.performance.now();

        // Start a timeout timer for the T1 response.
        _t1ToTimer = Timer(Duration(milliseconds: _t1TO), () {
          _handleResponse(isT1: true, response: 'timeout');
        });

        // Schedule T2 to appear after the specified SOA.
        _soaTimer = Timer(Duration(milliseconds: _currSOA), () {
          if (!mounted) return;
          // Show T2 (responses are always enabled when shown)
          setState(() { 
            _showT2 = true; 
            _canRespT2 = true;
          });

          // Schedule T2 timing logic for after the frame is rendered.
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (!mounted) return;
            _t2StartTime = html.window.performance.now();
            // Start T2 timeout timer immediately when T2 appears
            _t2ToTimer = Timer(Duration(milliseconds: _t2TO), () {
              _handleResponse(isT1: false, response: 'timeout');
            });
          });
        });
      });
    });
  }

  /// Handles a participant's response (or a timeout).
  ///
  /// [isT1]: True if the response is for T1, false for T2.
  /// [response]: The response code ('O', 'E', 'R', 'B') or 'timeout'.
  void _handleResponse({required bool isT1, required String response}) {
    if (!mounted) return;
    
    // Use high-precision timing with performance.now() for consistency
    final double now = html.window.performance.now();

    if (isT1) {
      if (!_canRespT1 || _t1Done) return; // Ignore if not allowed or already done
      _t1ToTimer?.cancel();
      _canRespT1 = false; // Disable further T1 responses.
      _t1Done = true;
      _t1Resp = response;
      _t1RespAbs = response == 'timeout' ? null : now;
      // Calculate RT, or set to null on timeout.
      _rt1 = response == 'timeout' || _t1StartTime == null ? null : now - _t1StartTime!;
    } else {
      if (!_showT2 || _t2Done) return; // Ignore if T2 not shown or already done
      _t2ToTimer?.cancel();
      _canRespT2 = false; // Disable further T2 responses.
      _t2Done = true;
      _t2Resp = response;
      _t2RespAbs = response == 'timeout' ? null : now;
      
      // Track if T2 was pressed before T1 completed
      if (!_t1Done && response != 'timeout') {
        _t2PressedWhileT1Pending = true;
      }
      
      _rt2 = response == 'timeout' || _t2StartTime == null ? null : now - _t2StartTime!;
    }

    // Once both responses are registered, end the trial.
    if (_t1Done && _t2Done) _finishTrial();
  }

  /// Called after both T1 and T2 responses are recorded.
  /// It logs data, handles transitions, and shows feedback for practice trials.
  void _finishTrial() {
    // Clean up any remaining timers for the trial.
    for (final t in [_fixTimer, _soaTimer, _t1ToTimer, _t2ToTimer]) { t?.cancel(); }
    
    final bool isPractice = _stage == PrpStage.practiceTrial;
    final int index = isPractice ? _pIdx : _tIdx;
    final Map<String, dynamic> trial = isPractice ? _practiceTrials[index] : _trials[index];

    // Determine if responses were correct.
    final bool t1Correct = _t1Resp != 'timeout' && _t1Resp == trial['t1_correct'];
    final bool t2Correct = _t2Resp != 'timeout' && _t2Resp == trial['t2_correct'];
    
    // Calculate response order
    String responseOrder = 'none';
    if (_t1RespAbs != null && _t2RespAbs != null) {
      responseOrder = (_t1RespAbs! <= _t2RespAbs!) ? 'T1→T2' : 'T2→T1';
    } else if (_t1RespAbs != null) {
      responseOrder = 'T1_only';
    } else if (_t2RespAbs != null) {
      responseOrder = 'T2_only';
    }
    
    // Calculate measured SOA (actual onset difference)
    double? measuredSoaMs = (_t1StartTime != null && _t2StartTime != null)
        ? (_t2StartTime! - _t1StartTime!)
        : null;
    
    // Helper function to convert absolute time to trial-relative time
    double? rel(double? abs) => (abs == null || _trialStartAbs == null) ? null : (abs - _trialStartAbs!);

    // For main trials, log the detailed trial data.
    if (!isPractice) {
      _log.add({
        // Common schema fields for Joint Bayesian (kept in wide format)
        'participant_id'               : _participantId,
        'task'                         : 'prp',
        'trial_index'                  : _tIdx,
        'block_index'                  : null,  // PRP doesn't have blocks
        
        // PRP-specific fields (wide format)
        'idx'                          : _tIdx,
        'soa_nominal_ms'               : trial['soa'],
        'soa_measured_ms'              : measuredSoaMs,
        
        't1_stim'                      : trial['t1_stim'],
        't1_correctResp'               : trial['t1_correct'],
        't1_resp'                      : _t1Resp ?? 'noResp',
        't1_correct'                   : t1Correct,
        't1_onset_ms'                  : _t1StartTime,  // absolute time
        't1_resp_ms'                   : _t1RespAbs,   // absolute time
        't1_rt_ms'                     : _rt1,
        't1_timeout'                   : _t1Resp == 'timeout',
        
        't2_stim'                      : trial['t2_stim'],
        't2_correctResp'               : trial['t2_correct'],
        't2_resp'                      : _t2Resp ?? 'noResp',
        't2_correct'                   : t2Correct,
        't2_onset_ms'                  : _t2StartTime,  // absolute time
        't2_resp_ms'                   : _t2RespAbs,    // absolute time
        't2_rt_ms'                     : _rt2,
        't2_timeout'                   : _t2Resp == 'timeout',
        
        'response_order'               : responseOrder,
        't2_pressed_while_t1_pending'  : _t2PressedWhileT1Pending,
      });
    }

    if (isPractice) {
      // For practice trials, prepare and show feedback.
      setState(() {
        _pFeedbackData = {
          't1_correct': t1Correct,
          't2_correct': t2Correct,
          't1_timeout': _t1Resp == 'timeout',
          't2_timeout': _t2Resp == 'timeout',
        };
        _stage = PrpStage.practiceFeedback;
      });
      // After feedback duration, move to the next practice trial.
      _fbTimer = Timer(Duration(milliseconds: _fbDur), () {
        if (!mounted) return;
        setState(() { _pIdx++; _stage = PrpStage.practiceTrial; });
        _startNextPracticeTrial();
      });
    } else {
      // For main trials, wait for the ITI then start the next trial.
      _itiTimer = Timer(Duration(milliseconds: _itiDur), () {
        if (!mounted) return;
        setState(() { _tIdx++; });
        _startNextMainTrial();
      });
    }
  }

  /// Initiates a rest break.
  void _startRestBreak() {
    setState(() {
      _stage = PrpStage.restBreak;
      _restCountdown = 10; // Set countdown for 10 seconds.
    });
    
    // A periodic timer that ticks every second to update the countdown.
    _restTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      
      setState(() {
        _restCountdown--;
      });
      
      // When countdown finishes, cancel the timer and continue the experiment.
      if (_restCountdown <= 0) {
        timer.cancel();
        _continueFromRest();
      }
    });
  }
  
  /// Continues the experiment after a rest break is over.
  void _continueFromRest() {
    _restTimer?.cancel();
    setState(() => _stage = PrpStage.mainTrial);
    
    // Instead of calling `_startNextMainTrial` (which would increment the trial
    // index again), we directly run the trial for the current index. `_tIdx`
    // was already incremented before the break began.
    if (_tIdx < _trials.length) {
      _runTrial(_trials[_tIdx]);
    } else {
      // This handles the edge case where the break was the very last thing.
      setState(() => _stage = PrpStage.result);
    }
  }

  // -------------------------------------------------------------------------
  // Section 6: Data Saving
  // -------------------------------------------------------------------------
  
  /// Calculates summary statistics, packages all data, and saves it to Firestore.
  /// Finally, it navigates back from the test page.
  Future<void> _saveAndExit() async {
    if (!mounted) return;

    // 1. Validate participant ID.
    if (_participantId.isEmpty || _participantId == 'unknown') {
      ScaffoldMessenger.of(context).showSnackBar(
        // "Error: Participant ID is not valid."
        const SnackBar(content: Text('오류: 참가자 ID가 유효하지 않습니다.')),
      );
      return;
    }

    // 2. Show a "Saving..." dialog to the user.
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Dialog(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children:[CircularProgressIndicator(), SizedBox(width:20), Text('저장 중...')], // "Saving..."
          ),
        ),
      ),
    );

    // 3. Prepare data for saving.
    final DateTime endTime = DateTime.now();
    final int durationSeconds = _experimentStartTime != null
        ? endTime.difference(_experimentStartTime!).inSeconds
        : 0;

    // Calculate mean T2 RT for each SOA condition.
    Map<String, int?> mrt2Soa = {};
    for (final soa in _soas) {
      final ss = _log.where((e) => e['soa_nominal_ms'] == soa);
      mrt2Soa['rt2_soa_$soa'] = _mrt(ss, 't2_rt_ms', 't2_correct')?.round();
    }

    // Long format conversion will be handled during analysis (not stored redundantly)

    try {
      // 4. Construct the main data object.
      final sessionId = "${DateTime.now().toIso8601String().replaceAll(':', '-').replaceAll('.', '-')}-$_participantId";
      final dataToSave = {
        // Session metadata for Joint Bayesian
        'session_id': sessionId,
        'app_version': '1.0.0',
        'device': 'web',
        'start_time': _experimentStartTime?.toIso8601String(),
        'end_time': endTime.toIso8601String(),
        'duration_seconds': durationSeconds,
        
        // Raw, trial-by-trial data (wide format only)
        'trialData'     : _log,
        
        // Summary statistics.
        'resultsSummary': {
          'n_trials'   : _log.length,
          'acc_t1'     : _acc(_log, 't1_correct'),
          'acc_t2'     : _acc(_log, 't2_correct'),
          'mrt_t1'     : _mrt(_log, 't1_rt_ms', 't1_correct')?.round(),
          'mrt_t2'     : _mrt(_log, 't2_rt_ms', 't2_correct')?.round(),
          ...mrt2Soa, // Include per-SOA RTs.
        },
        // Participant-specific configuration.
        'config': {
          'key_rev_t1': _reverseT1Keys,
          'key_rev_t2': _reverseT2Keys,
          'soas'      : _soas,
        },
        'submittedAt': FieldValue.serverTimestamp(),
      };

      // 5. Save data to Firestore.
      // The path is /participants/{participantId}/cognitive_tests/prp
      await FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('cognitive_tests')
          .doc('prp')
          .set(dataToSave, SetOptions(merge: true)); // `merge: true` prevents overwriting other docs.

      if (!mounted) return;

      // 6. Close the "Saving..." dialog.
      Navigator.of(context, rootNavigator: true).pop();

      // Short delay to ensure UI updates before popping the page.
      await Future.delayed(const Duration(milliseconds: 100));

      if (mounted) {
        print("PRP test completed for $_participantId. Returning to sequencer.");
        // 7. Return to the previous page (TestSequencerPage).
        Navigator.pop(context);
      }
    } catch (e) {
      if (mounted) {
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
    return WillPopScope(
      onWillPop: () async => false, // Block the user from using the back button.
      child: RawKeyboardListener(
        focusNode: _kbdFocus,
        onKey: _onKey,
        autofocus: true,  // Auto-focus on initial load
        child: Scaffold(
          appBar: AppBar(
            title: const Text('PRP 과제'), // "PRP Task"
            backgroundColor: Colors.white,
          ),
          backgroundColor: Colors.white,
          body: Column(
            children: [
              // Show a progress bar during practice and main trials.
              if (_progressVal != null) LinearProgressIndicator(value: _progressVal),
              Expanded(child: Center(child: _buildPrpStageUI())),
            ],
          ),
        ),
      ),
    );
  }

  /// Calculates the progress value for the linear progress indicator.
  double? get _progressVal {
    if (_stage == PrpStage.practiceTrial || _stage == PrpStage.practiceFeedback) {
      return (_pIdx + 1) / _practiceTrials.length;
    }
    if (_stage == PrpStage.mainTrial) {
      return _tIdx / _trials.length;
    }
    return null; // No progress bar on other screens.
  }

  /// The main UI router. Returns the widget corresponding to the current experiment stage.
  Widget _buildPrpStageUI() {
    switch (_stage) {
      case PrpStage.welcome:
        // "Welcome", "This is the PRP (Psychological Refractory Period) task.\nPlease respond to two stimuli in order, quickly."
        return _infoScreen('환영합니다', '본 검사는 PRP(심리적 불응기) 과제입니다.\n두 자극에 빠르게 순서대로 반응하세요.');
      case PrpStage.instructions:
        // "Task Instructions", "First a number will appear, then a colored square will appear after a short delay.\n\nPlease respond accurately and in order."
        return _infoScreen('과제 안내',
          '먼저 숫자가 표시되고, 잠시 후 색 사각형이 표시됩니다.\n\n'
          '순서대로 빠르고 정확하게 응답하세요.\n\n'
          '마우스는 사용할 수 없으며 키보드로만 응답해야 합니다.');
      case PrpStage.practiceInstructions:
        // "Practice Instructions", "You will now complete 10 practice trials."
        return _infoScreen('연습 안내', '연습 10번을 진행합니다.');
      case PrpStage.practiceTrial:
        return _trialUI();
      case PrpStage.practiceFeedback:
        return _practiceFeedbackUI();
      case PrpStage.mainTrialInstructions:
        // "Main Experiment", "You will now complete 120 trials. Press 'Next' when you are ready.\n\nA 10-second rest break will be given every 30 trials."
        return _infoScreen('본 시험', '총 120회 시행이 이어집니다. 준비되면 "다음"을 누르세요.\n\n30시행마다 10초의 휴식 시간이 주어집니다.');
      case PrpStage.mainTrial:
        return _trialUI();
      case PrpStage.restBreak:
        return _restBreakUI();
      case PrpStage.result:
        return _resultScreen();
    }
  }

  /// A reusable widget for displaying information screens (welcome, instructions).
  /// [title]: The screen title.
  /// [body]: The main text content.
  Widget _infoScreen(String title, String body) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(title, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
          const SizedBox(height: 16),
          Text(body, style: const TextStyle(fontSize: 16), textAlign: TextAlign.center),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: _onNext,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[200],
              foregroundColor: Colors.black,
            ),
            child: const Text('다음'),
          ), // "Next"
        ],
      ),
    );
  }

  /// Displays feedback after a practice trial (correct/incorrect/timeout).
  Widget _practiceFeedbackUI() {
    if (_pFeedbackData == null) return const SizedBox.shrink();
    // T1: "Timeout", "Correct", "Incorrect"
    String t1 = _pFeedbackData!['t1_timeout'] ? 'T1 시간초과' : _pFeedbackData!['t1_correct'] ? 'T1 정답' : 'T1 오답';
    // T2: "Timeout", "Correct", "Incorrect"
    String t2 = _pFeedbackData!['t2_timeout'] ? 'T2 시간초과' : _pFeedbackData!['t2_correct'] ? 'T2 정답' : 'T2 오답';
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(t1, style: TextStyle(fontSize: 24, color: _pFeedbackData!['t1_correct'] ? Colors.green : Colors.red)),
        const SizedBox(height: 10),
        Text(t2, style: TextStyle(fontSize: 24, color: _pFeedbackData!['t2_correct'] ? Colors.green : Colors.red)),
      ],
    );
  }

  /// The main UI for a single trial, showing stimuli and response buttons.
  Widget _trialUI() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // This container holds the stimuli (fixation, T1, T2).
        Container(
          height: 240,
          alignment: Alignment.center,
          child: _showFix ? const Text('+', style: TextStyle(fontSize: 60))
            : Column(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Visibility( // T1 stimulus (number)
                    visible: _showT1,
                    maintainAnimation: true, maintainState: true, maintainSize: true,
                    child: Text(_currT1?.toString() ?? '', style: const TextStyle(fontSize: 48, fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(height: 20),
                  Visibility( // T2 stimulus (colored square)
                    visible: _showT2,
                    maintainAnimation: true, maintainState: true, maintainSize: true,
                    child: Container(width: 80, height: 80, color: _currT2=='red'?Colors.red:Colors.blue),
                  ),
                ],
              ),
        ),
        const SizedBox(height: 24),
        const Text('T1(숫자)'), // "T1 (Number)"
        const SizedBox(height: 6),
        // Response buttons for T1
        IgnorePointer(
          ignoring: true, // 마우스/터치 완전 차단
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: (_showT1 && !_t1Done) ? () {} : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _aKeyPressed ? Colors.grey[400] : Colors.grey[200],
                  foregroundColor: Colors.black,
                ),
                child: Text('${_labelT1(_leftRespT1)} (A)'),
              ),
              const SizedBox(width: 20),
              ElevatedButton(
                onPressed: (_showT1 && !_t1Done) ? () {} : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _dKeyPressed ? Colors.grey[400] : Colors.grey[200],
                  foregroundColor: Colors.black,
                ),
                child: Text('${_labelT1(_rightRespT1)} (D)'),
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        const Text('T2(색)'), // "T2 (Color)"
        const SizedBox(height: 6),
        // Response buttons for T2
        IgnorePointer(
          ignoring: true, // 마우스/터치 완전 차단
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: (_showT2 && !_t2Done) ? () {} : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _leftKeyPressed ? Colors.grey[400] : Colors.grey[200],
                  foregroundColor: Colors.black,
                ),
                child: Text('${_labelT2(_leftRespT2)} (←)'),
              ),
              const SizedBox(width: 20),
              ElevatedButton(
                onPressed: (_showT2 && !_t2Done) ? () {} : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _rightKeyPressed ? Colors.grey[400] : Colors.grey[200],
                  foregroundColor: Colors.black,
                ),
                child: Text('${_labelT2(_rightRespT2)} (→)'),
              ),
            ],
          ),
        ),
      ],
    );
  }

  /// The final screen shown at the end of the experiment.
  Widget _resultScreen() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // "You have completed the task."
          const Text('고생하셨습니다.', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: _saveAndExit,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[200],
              foregroundColor: Colors.black,
            ),
            child: const Text('저장 후 종료'),
          ), // "Save and Exit"
        ],
      ),
    );
  }

  /// The UI for the rest break screen.
  Widget _restBreakUI() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            '휴식 시간', // "Rest Time"
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          Text(
            '$_restCountdown초 후 자동으로 계속됩니다', // "Resuming automatically in X seconds"
            style: const TextStyle(fontSize: 16),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          CircularProgressIndicator(
            // The indicator fills up as the countdown proceeds.
            value: (10 - _restCountdown) / 10,
          ),
        ],
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Section 8: Helper Methods
  // -------------------------------------------------------------------------
  
  /// Returns the button label for a T1 response code.
  /// [resp]: The response code ('O' or 'E').
  String _labelT1(String resp) {
    return resp == 'O' ? '홀수' : '짝수'; // "Odd" : "Even"
  }

  /// Returns the button label for a T2 response code.
  /// [resp]: The response code ('R' or 'B').
  String _labelT2(String resp) {
    return resp == 'R' ? '빨강' : '파랑'; // "Red" : "Blue"
  }

  /// Handles the "Next" button press on informational screens, advancing the experiment stage.
  void _onNext() {
    if (!mounted) return;
    setState(() {
      switch (_stage) {
        case PrpStage.welcome:              _stage = PrpStage.instructions; break;
        case PrpStage.instructions:         _stage = PrpStage.practiceInstructions; break;
        case PrpStage.practiceInstructions: _stage = PrpStage.practiceTrial; _pIdx=0; _startNextPracticeTrial(); break;
        case PrpStage.mainTrialInstructions: _stage = PrpStage.mainTrial; _tIdx=0; _startNextMainTrial(); break;
        default: break; // Do nothing on other stages
      }
    });
  }

  /// Calculates accuracy for a given condition.
  /// [d]: The list of trial data maps.
  /// [k]: The key for the correctness field (e.g., 't1_correct').
  double _acc(Iterable<Map<String,dynamic>> d, String k) {
    final n = d.length; if (n==0) return 0;
    return d.where((e) => e[k]==true).length * 100 / n;
  }

  /// Calculates the mean reaction time for correct trials.
  /// [d]: The list of trial data maps.
  /// [rtK]: The key for the reaction time field (e.g., 't1_rt').
  /// [accK]: The key for the correctness field (e.g., 't1_correct').
  double? _mrt(Iterable<Map<String,dynamic>> d, String rtK, String accK) {
    // Filter for correct trials with valid RTs > 0.
    final rts = d.where((e) => e[accK]==true && e[rtK]!=null && e[rtK]>0).map<double>((e) => e[rtK] as double);
    return rts.isNotEmpty ? rts.reduce((a,b)=>a+b)/rts.length : null;
  }
}
