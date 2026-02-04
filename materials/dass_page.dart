import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class DASS21Page extends StatefulWidget {
  const DASS21Page({Key? key}) : super(key: key);

  @override
  State<DASS21Page> createState() => _DASS21PageState();
}

class _DASS21PageState extends State<DASS21Page> {
  late String _participantId;
  bool _isLoading = true;
  DateTime? _startTime;

  final List<Map<String, dynamic>> _questions = [
    {"questionText": "1. 나는 안정을 취하기 힘들었다.", "scale": "S"},
    {"questionText": "2. 입이 바싹 마르는 느낌이 들었다.", "scale": "A"},
    {"questionText": "3. 어떤 것에도 긍정적인 감정을 느낄 수가 없었다.", "scale": "D"},
    {"questionText": "4. 숨쉬기가 곤란한 적이 있었다 (심하게 호흡이 가쁘거나 가만히 있을 때도 호흡 곤란이 있었다).", "scale": "A"},
    {"questionText": "5. 무엇인가를 시작하는 것이 어려웠다.", "scale": "D"},
    {"questionText": "6. 어떤 상황에 과잉 반응을 보이는 경향이 있었다.", "scale": "S"},
    {"questionText": "7. 몸이 떨리는 것을 느꼈다(예: 손 떨림).", "scale": "A"},
    {"questionText": "8. 모든 일에 신경을 너무 많이 쓴다고 느꼈다.", "scale": "S"},
    {"questionText": "9. 내가 너무 당황해서 웃음거리가 될까 봐 걱정했다.", "scale": "A"},
    {"questionText": "10. 나는 기대할 것이 아무것도 없다는 생각이 들었다.", "scale": "D"},
    {"questionText": "11. 자꾸 초조해졌다.", "scale": "S"},
    {"questionText": "12. 나는 진정하는 것이 어려웠다.", "scale": "S"},
    {"questionText": "13. 기운이 처지고 우울했다.", "scale": "D"},
    {"questionText": "14. 내가 하는 일에 방해가 되는 것을 용납할 수 없었다.", "scale": "S"},
    {"questionText": "15. 내 자신이 심한 불안상태까지 도달했음을 느꼈다.", "scale": "A"},
    {"questionText": "16. 어떤 것에도 몰두 할 수가 없었다.", "scale": "D"},
    {"questionText": "17. 나는 사람으로서 가치가 없다고 느꼈다.", "scale": "D"},
    {"questionText": "18. 내가 꽤 신경질적이라고 느꼈다.", "scale": "S"},
    {"questionText": "19. 가만히 있을 때에도 심장이 두근거리는 것이 느껴졌다 (예: 심장이 심하게 빨리 뛰는 느낌, 불규칙한 심장 박동).", "scale": "A"},
    {"questionText": "20. 아무 이유 없이 무서움을 느꼈다.", "scale": "A"},
    {"questionText": "21. 산다는 것이 의미가 없다는 생각이 들었다.", "scale": "D"},
  ];

  final List<String> _options = [
    "전혀 해당되지 않음",
    "약간 또는 가끔 해당됨",
    "상당히 또는 자주 해당됨",
    "매우 많이 또는 거의 대부분 해당됨"
  ];

  final List<int?> _responses = List.filled(21, null);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
       if (mounted) {
          _initializeAndCheckCompletion();
       }
    });
  }

  Future<void> _initializeAndCheckCompletion() async {
    if (ModalRoute.of(context)!.settings.arguments != null) {
      final arg = ModalRoute.of(context)!.settings.arguments;
      if (arg is String) {
        _participantId = arg;
        print("DASSPage received participantId: $_participantId");
        await _checkDassCompletionAndStartTime();
      } else {
        print("Warning: Received non-String argument for participantId. Type: ${arg.runtimeType}");
        _participantId = "unknown_participant_error";
        _handleIdError('오류: 사용자 ID를 받지 못했습니다.');
      }
    } else {
      print("Warning: No arguments received for participantId.");
      _participantId = "unknown_participant_no_args";
      _handleIdError('오류: 사용자 ID가 전달되지 않았습니다.');
    }
     if (mounted) {
        setState(() => _isLoading = false);
     }
  }

  void _handleIdError(String message) {
      if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(message)),
          );
          setState(() => _isLoading = false);
      }
  }

  Future<void> _checkDassCompletionAndStartTime() async {
    if (_participantId.startsWith("unknown_participant")) {
      if (mounted) setState(() => _isLoading = false);
      return;
    }

    try {
      final docRef = FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('surveys')
          .doc('dass');
      final docSnapshot = await docRef.get();

      if (docSnapshot.exists) {
        print("DASS-21 survey document already exists for $_participantId. Navigating to /cognitive_test_intro.");
        if (mounted) {
             Navigator.pushReplacementNamed(context, '/cognitive_test_intro', arguments: _participantId);
        }
      } else {
        _startTime = DateTime.now();
        await docRef.set({
          'start_time': _startTime,
          'end_time': null,
          'duration_seconds': null,
          'scores': null,
          'responses': null,
        });
        print("DASS-21 survey started for $_participantId at $_startTime. Initial document created at .../surveys/dass");
      }
    } catch (e) {
      print("Error checking DASS-21 completion or starting survey time log: $e");
      if (mounted) {
           ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('DASS 설문 상태 확인 또는 시작 시간 기록 중 오류 발생: $e')),
           );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text("DASS-21 척도 (우울, 불안, 스트레스)")),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text("DASS-21 척도 (우울, 불안, 스트레스)"),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildIntroduction(),
            const SizedBox(height: 20),
            const Divider(),
            const SizedBox(height: 20),
            ...List.generate(_questions.length, (index) {
              return _buildQuestionCard(index);
            }),
            const SizedBox(height: 30),
            Center(
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                    textStyle: const TextStyle(fontSize: 16)),
                onPressed: _isLoading ? null : _submitDassSurvey,
                child: _isLoading
                    ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3,))
                    : const Text("제출하기"),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildIntroduction() {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.grey.shade300)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "지난 한 주 동안, 아래의 문항이 귀하에게 얼마큼 해당되었는지 0, 1, 2, 3번에 동그라미 표시해 주십시오. 정답이 있는 것이 아니므로 오래 생각하지 마시고 답해 주시기 바랍니다.",
            style: TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
          ),
          const SizedBox(height: 15),
          Padding(
            padding: const EdgeInsets.only(left: 8.0),
            child: Text(
              _options.asMap().entries.map((e) => "${e.key}. ${e.value}").join("  "),
              style: const TextStyle(
                  fontSize: 15,
                  height: 1.6,
                  color: Colors.black87,
                  fontWeight: FontWeight.w500),
            ),
          ),
          const SizedBox(height: 15),
          const Text(
            "모든 문항에 빠짐없이 응답해 주세요.",
            style: TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
          ),
          SizedBox(height: 10),
          Text(
            "여러분의 응답은 익명으로 처리되며, 연구 목적 이외에는 사용되지 않습니다. 설문 도중 불편함을 느끼실 경우 언제든 중단하거나 연구자에게 문의하실 수 있습니다.",
            style: TextStyle(fontSize: 14, height: 1.4, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildQuestionCard(int index) {
    final question = _questions[index];
    final questionText = question["questionText"] as String;
    final response = _responses[index];

    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.15),
              spreadRadius: 1,
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ]),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              questionText,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8.0,
              runSpacing: 0.0,
              alignment: WrapAlignment.spaceAround,
              children: _options.asMap().entries.map((entry) {
                final optIndex = entry.key;
                final optLabel = entry.value;
                final optValue = optIndex;
                return InkWell(
                  onTap: () {
                    setState(() {
                      _responses[index] = optValue;
                    });
                  },
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Radio<int>(
                        value: optValue,
                        groupValue: response,
                        onChanged: (value) {
                          setState(() {
                            _responses[index] = value;
                          });
                        },
                      ),
                      Text(optLabel),
                    ],
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Map<String, int> _calculateScores() {
    Map<String, int> scores = {'D': 0, 'A': 0, 'S': 0};
    for (int i = 0; i < _questions.length; i++) {
      String scale = _questions[i]['scale'] as String;
      int responseValue = _responses[i]!;
      scores[scale] = scores[scale]! + responseValue;
    }
    scores['D'] = scores['D']! * 2;
    scores['A'] = scores['A']! * 2;
    scores['S'] = scores['S']! * 2;
    return scores;
  }

  Future<void> _submitDassSurvey() async {
    if (_responses.any((r) => r == null)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("모든 질문에 답변해주세요.")),
      );
      return;
    }

    if (_participantId.startsWith("unknown_participant")) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("오류: 사용자 ID가 유효하지 않아 제출할 수 없습니다.")),
      );
      return;
    }

    if (_startTime == null) {
      print("Error: _startTime is null during DASS-21 survey submission. Cannot accurately record time.");
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("오류: 설문 시작 시간을 기록하지 못했습니다. 다시 시도해주세요.")),
      );
      setState(() => _isLoading = false);
      return;
    }

    setState(() => _isLoading = true);

    try {
      final endTime = DateTime.now();
      final duration = endTime.difference(_startTime!);

      final dassScores = _calculateScores();

      await FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('surveys')
          .doc('dass')
          .update({
        'end_time': endTime,
        'duration_seconds': duration.inSeconds,
        'scores': dassScores,
        'responses': _responses,
      });

      print("DASS-21 survey submitted for $_participantId with scores: D-${dassScores['D']}, A-${dassScores['A']}, S-${dassScores['S']}. Times recorded in .../surveys/dass doc.");

      if (mounted) {
        Navigator.pushReplacementNamed(context, '/cognitive_test_intro', arguments: _participantId);
      }
    } catch (e) {
      print("Error submitting DASS-21 survey: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("DASS-21 제출 중 오류 발생: $e")),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }
}