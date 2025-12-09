import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class UCLAPage extends StatefulWidget {
  const UCLAPage({Key? key}) : super(key: key);

  @override
  State<UCLAPage> createState() => _UCLAPageState();
}

class _UCLAPageState extends State<UCLAPage> {
  late String _participantId;
  bool _isLoading = true;
  DateTime? _startTime;

  final List<Map<String, dynamic>> _questions = [
    {"questionText": "1. 얼마나 자주 주변 사람들과 잘 통한다고 느끼십니까?", "reverse": true},
    {"questionText": "2. 얼마나 자주 사람들과의 교제가 부족하다고 느끼십니까?", "reverse": false},
    {"questionText": "3. 얼마나 자주 도움을 청할 사람이 아무도 없다고 느끼십니까?", "reverse": false},
    {"questionText": "4. 얼마나 자주 혼자라고 느끼십니까?", "reverse": false},
    {"questionText": "5. 얼마나 자주 친구들 모임에 속해 있다고 느끼십니까?", "reverse": true},
    {"questionText": "6. 얼마나 자주 당신 주위 사람들과 비슷한 점이 많다고 느끼십니까?", "reverse": true},
    {"questionText": "7. 얼마나 자주 당신이 더 이상 아무하고도 가깝지 않다고 느끼십니까?", "reverse": false},
    {"questionText": "8. 얼마나 자주 당신의 흥미와 생각들이 주변 사람과 나누어지지 않는다고 느끼십니까?", "reverse": false},
    {"questionText": "9. 얼마나 자주 자신이 적극적이고 호의적이라고 느끼십니까?", "reverse": true},
    {"questionText": "10. 얼마나 자주 사람들과 가깝다고 느끼십니까?", "reverse": true},
    {"questionText": "11. 얼마나 자주 혼자 남겨졌다고 느끼십니까?", "reverse": false},
    {"questionText": "12. 얼마나 자주 다른 사람들과의 관계가 의미 없다고 느끼십니까?", "reverse": false},
    {"questionText": "13. 얼마나 자주 당신을 진정으로 아는 사람이 아무도 없다고 느끼십니까?", "reverse": false},
    {"questionText": "14. 얼마나 자주 다른 사람들로부터 고립되어 있다고 느끼십니까?", "reverse": false},
    {"questionText": "15. 얼마나 자주 당신이 원할 때에 함께 있어 줄 사람을 찾을 수 있다고 느끼십니까?", "reverse": true},
    {"questionText": "16. 얼마나 자주 당신을 정말 이해해주는 사람들이 있다고 느끼십니까?", "reverse": true},
    {"questionText": "17. 얼마나 자주 수줍음을 느끼십니까?", "reverse": false},
    {"questionText": "18. 얼마나 자주 사람들이 당신과 진정으로 함께 있지 않고 그저 주위에 있는 것이라고 느끼십니까?", "reverse": false},
    {"questionText": "19. 얼마나 자주 당신이 얘기할 수 있는 사람들이 주변에 있다고 생각합니까?", "reverse": true},
    {"questionText": "20. 얼마나 자주 당신이 도움을 청할 수 있는 사람들이 있다고 느끼십니까?", "reverse": true},
  ];

  final List<int?> _responses = List.filled(20, null);

  final List<String> _options = [
    "전혀 그렇지 않았다",
    "거의 그렇지 않았다",
    "가끔 그랬다",
    "자주 그랬다"
  ];

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
        print("UCLAPage received participantId: $_participantId");
        await _checkUclaCompletionAndStartTime();
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
    }
  }

  Future<void> _checkUclaCompletionAndStartTime() async {
    if (_participantId.startsWith("unknown_participant")) {
      if (mounted) setState(() => _isLoading = false);
      return;
    }

    try {
      final docRef = FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('surveys')
          .doc('ucla');
      final docSnapshot = await docRef.get();

      if (docSnapshot.exists) {
        print("UCLA survey document already exists for $_participantId. Navigating to /dass.");
        if (mounted) {
             Navigator.pushReplacementNamed(context, '/dass', arguments: _participantId);
        }
      } else {
        _startTime = DateTime.now();
        await docRef.set({
          'start_time': _startTime,
          'end_time': null,
          'duration_seconds': null,
          'score': null,
          'responses': null,
        });
        print("UCLA survey started for $_participantId at $_startTime. Initial document created at .../surveys/ucla");
      }
    } catch (e) {
      print("Error checking UCLA completion or starting survey time log: $e");
      if (mounted) {
           ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('설문 상태 확인 또는 시작 시간 기록 중 오류 발생: $e')),
           );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text("UCLA 외로움 척도(Version 3)")),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("UCLA 외로움 척도(Version 3)"),
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
                    textStyle: const TextStyle(fontSize: 16)
                ),
                onPressed: _isLoading ? null : _submitSurvey,
                child: _isLoading ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3,)) : const Text("제출하기"),
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
          border: Border.all(color: Colors.grey.shade300)
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "이 설문은 지난 일주일(7일) 동안 여러분이 경험한 사회적 관계와 감정(외로움)에 대해 묻고자 합니다. 각 문항을 읽고, 아래 네 가지 보기 중 가장 가까운 빈도를 선택해 주세요.",
            style: TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
          ),
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.only(left: 8.0),
            child: Text(
              _options.asMap().entries.map((e) => "${e.key + 1}. ${e.value}").join("  "),
              style: const TextStyle(fontSize: 15, height: 1.6, color: Colors.black87, fontWeight: FontWeight.w500),
            ),
          ),
          const SizedBox(height: 15),
          const Text(
            "정답은 없습니다. 느끼신 그대로 솔직하고 신중하게 응답해 주시면 감사하겠습니다. 모든 문항에 빠짐없이 응답해 주세요.",
            style: TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
          ),
          const SizedBox(height: 10),
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
          ]
      ),
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
                final optValue = optIndex + 1;
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

  Future<void> _submitSurvey() async {
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
      print("Error: _startTime is null during UCLA survey submission. Cannot accurately record time.");
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

      int score = 0;
      for (int i = 0; i < _questions.length; i++) {
        bool reversed = _questions[i]["reverse"] as bool;
        int responseValue = _responses[i]!;
        if (reversed) {
          score += (5 - responseValue);
        } else {
          score += responseValue;
        }
      }

      await FirebaseFirestore.instance
          .collection('participants')
          .doc(_participantId)
          .collection('surveys')
          .doc('ucla')
          .update({
        'end_time': endTime,
        'duration_seconds': duration.inSeconds,
        'score': score,
        'responses': _responses,
      });

      print("UCLA survey submitted for $_participantId with score: $score. Times recorded in .../surveys/ucla doc.");

      if (mounted) {
        Navigator.pushReplacementNamed(context, '/dass', arguments: _participantId);
      }
    } catch (e) {
      print("Error submitting UCLA survey: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("제출 중 오류 발생: $e")),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }
}