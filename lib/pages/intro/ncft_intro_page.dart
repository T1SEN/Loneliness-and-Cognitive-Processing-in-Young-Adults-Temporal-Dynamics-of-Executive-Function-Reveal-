import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'end_page.dart';
import 'test_sequencer_page.dart';

enum CompletionStatus { loading, notCompleted, completed, error }

class NCFTIntroPage extends StatefulWidget {
  const NCFTIntroPage({Key? key}) : super(key: key);

  @override
  State<NCFTIntroPage> createState() => _NCFTIntroPageState();
}

class _NCFTIntroPageState extends State<NCFTIntroPage> {
  late String _participantId;
  bool _isLoading = true;
  List<String> _testOrder = [];

  CompletionStatus _stroopStatus = CompletionStatus.loading;
  CompletionStatus _wcstStatus = CompletionStatus.loading;
  CompletionStatus _prpStatus = CompletionStatus.loading;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _initializeAndCheckCompletionStatus();
      }
    });
  }

  Future<void> _initializeAndCheckCompletionStatus() async {
    if (ModalRoute.of(context)!.settings.arguments != null) {
      final arg = ModalRoute.of(context)!.settings.arguments;
      if (arg is String && arg.isNotEmpty && !arg.startsWith("unknown")) {
        _participantId = arg;
        print("NCFTIntroPage received participantId: $_participantId");

        if (mounted) {
          setState(() {
            _testOrder = _generateRandomTestOrder();
            print("Randomized test order: $_testOrder");
          });
        }
        await _fetchCompletionStatus();
      } else {
        print("Warning: Invalid participantId received. Arg: $arg");
        _participantId = "invalid_participant_id_error";
        _handleIdError('오류: 유효하지 않은 사용자 ID를 받았습니다.');
        setState(() {
           _stroopStatus = CompletionStatus.error;
           _wcstStatus = CompletionStatus.error;
           _prpStatus = CompletionStatus.error;
           _isLoading = false;
        });
        return;
      }
    } else {
      print("Warning: No arguments received for participantId.");
      _participantId = "no_participant_id_error";
      _handleIdError('오류: 사용자 ID가 전달되지 않았습니다.');
        setState(() {
           _stroopStatus = CompletionStatus.error;
           _wcstStatus = CompletionStatus.error;
           _prpStatus = CompletionStatus.error;
           _isLoading = false;
        });
      return;
    }
     if (mounted) {
        setState(() => _isLoading = false);
     }
  }

  List<String> _generateRandomTestOrder() {
    // Stroop과 PRP만 랜덤하게 섞고, WCST는 마지막에 고정
    List<String> shuffleableTests = ['/stroop', '/prp'];
    shuffleableTests.shuffle();
    
    // 섞인 Stroop/PRP 뒤에 WCST 추가
    return [...shuffleableTests, '/wcst'];
  }

  Future<void> _fetchCompletionStatus() async {
    if (_participantId.contains("_error")) return;

    final testRef = FirebaseFirestore.instance
        .collection('participants')
        .doc(_participantId)
        .collection('cognitive_tests');

    try {
      final stroopDoc = await testRef.doc('stroop').get();
      final wcstDoc = await testRef.doc('wcst').get();
      final prpDoc = await testRef.doc('prp').get();

      if (mounted) {
         setState(() {
            _stroopStatus = stroopDoc.exists ? CompletionStatus.completed : CompletionStatus.notCompleted;
            _wcstStatus = wcstDoc.exists ? CompletionStatus.completed : CompletionStatus.notCompleted;
            _prpStatus = prpDoc.exists ? CompletionStatus.completed : CompletionStatus.notCompleted;
         });
      }
    } catch (e) {
      print("Error fetching completion status: $e");
      if (mounted) {
         setState(() {
            _stroopStatus = CompletionStatus.error;
            _wcstStatus = CompletionStatus.error;
            _prpStatus = CompletionStatus.error;
         });
         ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('오류: 완료 상태를 불러오지 못했습니다. $e')),
         );
      }
    }
  }

  void _handleIdError(String message) {
      if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(message)),
          );
      }
  }

  @override
  Widget build(BuildContext context) {
     if (_isLoading) {
        return Scaffold(
            appBar: AppBar(title: const Text("신경인지기능검사 안내")),
            body: const Center(child: CircularProgressIndicator()),
        );
     }
     if (_participantId.contains("_error")) {
         return Scaffold(
            appBar: AppBar(title: const Text("오류")),
            body: const Center(child: Text("오류가 발생하여 진행할 수 없습니다.\n앱을 다시 시작해주세요.", textAlign: TextAlign.center)),
         );
     }

    return Scaffold(
      appBar: AppBar(
        title: const Text("신경인지기능검사 안내"),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeaderSection(),
            const SizedBox(height: 30),
            _buildStroopSection(),
            const SizedBox(height: 25),
            const Divider(),
            const SizedBox(height: 25),
            _buildWCSTSection(),
            const SizedBox(height: 25),
            const Divider(),
            const SizedBox(height: 25),
            _buildPRPSection(),
            const SizedBox(height: 30),
            const Divider(),
            const SizedBox(height: 30),
            _buildImportantNotesSection(),
            const SizedBox(height: 40),
            _buildStartButton(),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildHeaderSection() {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 3,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [

          Text(
            "지금부터 세 가지 신경인지기능검사를 진행하게 됩니다. 각 검사는 인지 능력의 서로 다른 중요한 측면(주의력, 실행기능, 정보처리 속도 등)을 평가합니다.",
            style: TextStyle(fontSize: 16, height: 1.5, color: Colors.black87),
          ),
          SizedBox(height: 12),
          Text(
            "각 검사 시작 전, 자세한 수행 방법이 안내될 것입니다. 모든 검사는 정확하고 신속한 반응이 중요하므로, 안내를 주의 깊게 읽고 최대한 집중하여 참여해 주시기 바랍니다.",
            style: TextStyle(fontSize: 16, height: 1.5, color: Colors.black87),
          ),
        ],
      ),
    );
  }

  Widget _buildTestSectionCard({required String title, required CompletionStatus status, required List<Widget> children}) {
      return Container(
        margin: const EdgeInsets.only(bottom: 5),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: _getStatusColor(status).withOpacity(0.5)),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.15),
              spreadRadius: 1,
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                 mainAxisAlignment: MainAxisAlignment.spaceBetween,
                 children: [
                    Expanded(
                      child: Text(
                         title,
                         style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.black87,
                         ),
                      ),
                    ),
                    Text(
                       _getStatusText(status),
                       style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: _getStatusColor(status),
                       ),
                    ),
                 ],
              ),
              const SizedBox(height: 15),
              ...children,
            ],
          ),
        ),
      );
  }

  String _getStatusText(CompletionStatus status) {
    switch (status) {
      case CompletionStatus.loading:
        return " (확인 중...)";
      case CompletionStatus.notCompleted:
        return " (미완료)";
      case CompletionStatus.completed:
        return " (완료)";
      case CompletionStatus.error:
        return " (오류)";
    }
  }

  Color _getStatusColor(CompletionStatus status) {
      switch (status) {
        case CompletionStatus.loading:
           return Colors.grey;
        case CompletionStatus.notCompleted:
           return Colors.orange.shade700;
        case CompletionStatus.completed:
           return Colors.green.shade700;
        case CompletionStatus.error:
           return Colors.red.shade700;
      }
  }

  Widget _buildStroopSection() {
    return _buildTestSectionCard(
      title: "1. 스트룹 검사 (Stroop Test)",
      status: _stroopStatus,
      children: [
        _buildSectionContent(
          "글자의 의미와 글자의 색깔이 주어집니다. 제시되는 글자의 '색깔'과 일치하는 버튼을 최대한 빠르고 정확하게 누르는 검사입니다."
          " 예를 들어, '빨강'이라는 단어가 <파란색>으로 쓰여 있다면, '파란색' 버튼을 선택해야 합니다.",
        ),
        const SizedBox(height: 10),
        _buildSectionContent(
          "이 검사는 불필요한 정보를 억제하고 목표 정보에 집중하는 능력(선택적 주의력, 간섭 통제)을 측정합니다.",
        ),
        const SizedBox(height: 10),
        _buildHighlightContent(
           "※ 중요: 이 검사는 색상을 정확히 구분해야 하므로, 색맹 또는 색약이 있으신 경우 검사 진행이 어렵습니다.",
           icon: Icons.color_lens,
           color: Colors.orange.shade700
        ),
         const SizedBox(height: 5),
        _buildSectionContent(
          "예상 소요 시간: 약 5분",
           icon: Icons.timer_outlined
        ),
      ],
    );
  }

  Widget _buildWCSTSection() {
    return _buildTestSectionCard(
      title: "2. 위스콘신 카드 분류 검사 (WCST)",
      status: _wcstStatus,
      children: [
        _buildSectionContent(
          "화면 상단에 4장의 기준 카드와 하단에 1장의 반응 카드가 제시됩니다. 특정 규칙(색깔, 모양, 또는 개수)에 따라 반응 카드를 4개의 기준 카드 중 하나와 짝지어야 합니다.",
        ),
         const SizedBox(height: 10),
        _buildSectionContent(
          "규칙은 검사 도중에 예고 없이 변경됩니다. 여러분은 컴퓨터가 제공하는 피드백('정답' 또는 '오답')을 통해 현재 적용되는 새로운 규칙을 찾아내고 적용해야 합니다.",
        ),
        const SizedBox(height: 10),
        _buildSectionContent(
          "이 검사는 변화하는 상황에 맞춰 사고방식을 유연하게 전환하고, 문제를 해결하며, 추상적인 규칙을 학습하는 능력(실행 기능, 인지 유연성)을 측정합니다.",
        ),
         const SizedBox(height: 10),
        _buildSectionContent(
          "예상 소요 시간: 약 10-15분",
           icon: Icons.timer_outlined
        ),
      ],
    );
  }

  Widget _buildPRPSection() {
    return _buildTestSectionCard(
      title: "3. 심리적 불응기 검사 (PRP)",
      status: _prpStatus,
      children: [
         _buildSectionContent(
            "이 검사에서는 짧은 시간 간격을 두고 두 가지 과제가 연속적으로 제시됩니다. 각 과제에 대해 최대한 빠르고 정확하게 반응해야 합니다."),
        const SizedBox(height: 10),
        _buildSectionContent(
          "두 번째 과제에 대한 반응은 첫 번째 과제 처리 때문에 지연될 수 있습니다. 이 검사는 여러 자극을 동시에 처리하는 능력과 주의력 자원의 한계를 측정합니다(주의력 병목현상, 정보처리 속도).",
        ),

        const SizedBox(height: 5),
        _buildSectionContent(
          "예상 소요 시간: 약 10분",
           icon: Icons.timer_outlined
        ),
      ],
    );
  }

  Widget _buildImportantNotesSection() {
    return Container(
      padding: const EdgeInsets.all(16.0),
       decoration: BoxDecoration(
        color: Colors.yellow.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.orange.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "검사 진행 시 유의사항",
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 15),
          _buildSectionContent(
            "• 환경: 주변 소음이 없고 방해받지 않는 조용한 환경에서 검사를 진행해 주세요.",
            icon: Icons.volume_off_outlined,
          ),
          const SizedBox(height: 10),
          _buildSectionContent(
            "• 집중: 검사 중에는 최대한 집중력을 유지해 주시기 바랍니다. 각 문항에 신중하고 빠르게 반응해 주세요.",
            icon: Icons.center_focus_strong_outlined,
          ),
          const SizedBox(height: 10),
          _buildSectionContent(
            "• 순서: 검사 순서는 참가자마다 다르게 제시될 수 있습니다 (Counterbalancing). 이는 연구 결과의 편향을 줄이기 위함이니 참고 부탁드립니다.",
            icon: Icons.shuffle_outlined,
          ),
           const SizedBox(height: 10),
          _buildSectionContent(
            "• 휴식: 각 검사 사이에는 짧은 휴식 안내가 있을 수 있습니다. 충분히 준비된 후 다음 검사를 시작해 주세요.",
            icon: Icons.pause_circle_outline,
          ),
          const SizedBox(height: 10),
           _buildSectionContent(
            "• 중단: 검사 도중 부득이하게 중단해야 할 경우, 진행 상황이 저장되지 않을 수 있습니다. 가능한 한 모든 검사를 한 번에 완료해 주시는 것이 좋습니다.",
            icon: Icons.warning_amber_outlined,
            color: Colors.orange.shade800
          ),
        ],
      ),
    );
  }

  Widget _buildSectionContent(String content, {IconData? icon, Color? color}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
         if (icon != null) ...[
          Icon(icon, size: 18, color: color ?? Colors.black54),
          const SizedBox(width: 8),
        ],
        Expanded(
          child: Text(
            content,
            style: TextStyle(fontSize: 15.5, height: 1.6, color: color ?? Colors.black87),
          ),
        ),
      ],
    );
  }

   Widget _buildHighlightContent(String content, {required IconData icon, Color? color}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: color ?? Colors.red.shade700),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            content,
            style: TextStyle(
                fontSize: 15.5,
                height: 1.6,
                color: color ?? Colors.red.shade700,
                fontWeight: FontWeight.w500
            ),
          ),
        ),
      ],
    );
  }

  void _showStartDialog() async {
    final confirmed = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        title: const Text(
          "검사 진행 안내",
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        content: Container(
          constraints: const BoxConstraints(maxWidth: 400),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.warning_amber, color: Colors.orange.shade700, size: 20),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text(
                      "검사 도중에는 중단할 수 없습니다.",
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Icon(Icons.timer, color: Colors.blue.shade700, size: 20),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text("각 검사 사이에는 2분 휴식이 자동으로 제공됩니다."),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Icon(Icons.psychology, color: Colors.green.shade700, size: 20),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text("총 3개의 검사가 랜덤 순서로 진행됩니다."),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.desktop_windows_outlined, color: Colors.blueGrey.shade700, size: 20),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text("조용하고 소음 없는 장소에서 컴퓨터로 진행해 주세요."),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              const Text(
                "검사를 시작하시겠습니까?",
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            child: const Text("취소"),
            onPressed: () => Navigator.pop(context, false),
          ),
          ElevatedButton(
            child: const Text("시작"),
            onPressed: () => Navigator.pop(context, true),
          ),
        ],
      ),
    );

    if (confirmed == true && mounted) {
      final order = _generateRandomTestOrder();
      print("Starting cognitive test sequence with order: $order");
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => TestSequencerPage(
            participantId: _participantId,
            testOrder: order,
          ),
        ),
      );
    }
  }

  Widget _buildStartButton() {
    bool allCompleted = _stroopStatus == CompletionStatus.completed &&
                       _wcstStatus == CompletionStatus.completed &&
                       _prpStatus == CompletionStatus.completed;
    bool isLoading = _stroopStatus == CompletionStatus.loading ||
                      _wcstStatus == CompletionStatus.loading ||
                      _prpStatus == CompletionStatus.loading;
     bool hasError = _stroopStatus == CompletionStatus.error ||
                      _wcstStatus == CompletionStatus.error ||
                      _prpStatus == CompletionStatus.error ||
                      _participantId.contains("_error");

    String buttonText = "인지 검사 시작하기";
    VoidCallback? onPressedAction = _showStartDialog;

    if (isLoading) {
       buttonText = "완료 상태 확인 중...";
       onPressedAction = null;
    } else if (hasError) {
       buttonText = "오류: 검사 진행 불가";
       onPressedAction = null;
    } else if (allCompleted) {
      buttonText = "모든 검사 완료됨";
      onPressedAction = () {
         print("All cognitive tests completed for $_participantId. Navigating to end page.");
         Navigator.pushReplacementNamed(context, '/end_page');
      };
    }

    return Center(
      child: ElevatedButton(
        onPressed: onPressedAction,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 50, vertical: 18),
          textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          backgroundColor: onPressedAction == null ? Colors.grey : null,
        ),
        child: Text(buttonText),
      ),
    );
  }
}