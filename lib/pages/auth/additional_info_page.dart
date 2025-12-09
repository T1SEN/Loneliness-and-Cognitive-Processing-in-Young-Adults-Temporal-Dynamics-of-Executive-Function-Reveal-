import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AdditionalInfoPage extends StatefulWidget {
  const AdditionalInfoPage({Key? key}) : super(key: key);

  @override
  _AdditionalInfoPageState createState() => _AdditionalInfoPageState();
}

class _AdditionalInfoPageState extends State<AdditionalInfoPage> {
  final _formKey = GlobalKey<FormState>();
  String? _selectedGender;
  String? _selectedEducation;
  String? _courseName;
  String? _classSection;
  String? _professorName;
  final _otherNotesController = TextEditingController();
  bool _isLoading = false;
  String? _errorMessage;

  final List<String> _genders = ['남성', '여성'];
  final List<String> _educationLevels = [
    '대학교 재학',
    '대학원 재학',
    '기타'
  ];
  final List<String> _courseNames = ['심리학개론', '공존과협력의심리', '현대사회와정신건강', '인지심리학', '행복심리학', '발달심리학', '학습심리학', '사회심리학', '언어및사고심리학', '상담심리학', '임상심리학', '중독심리학', '참여하지 않음'];
  final List<String> _classSections = ['1', '2', '3', '4', '5', '6', '참여하지 않음'];
  final List<String> _professorNames = ['양동옥 교수님', '위경선 교수님', '김서홍 교수님', '이세라 교수님', '황석현 교수님', '이혜지 교수님', '김천수 교수님', '김상엽 교수님', '전성은 교수님', '한현섭 교수님', '염영미 교수님', '참여하지 않음'];

  @override
  void dispose() {
    _otherNotesController.dispose();
    super.dispose();
  }

  Future<void> _submitAdditionalInfo(String userUid) async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      await FirebaseFirestore.instance.collection('participants').doc(userUid).update({
        'gender': _selectedGender,
        'education': _selectedEducation,
        'courseName': _courseName,
        'classSection': _classSection,
        'professorName': _professorName,
        'otherNotes': _otherNotesController.text.isNotEmpty ? _otherNotesController.text : null,
      });

      if (mounted) {
        Navigator.of(context).pushReplacementNamed('/ucla', arguments: userUid);
      }
    } catch (e) {
      setState(() {
        _errorMessage = '정보 저장 중 오류가 발생했습니다: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final userUid = ModalRoute.of(context)?.settings.arguments as String?;

    if (userUid == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('오류')),
        body: const Center(child: Text('사용자 ID를 전달받지 못했습니다. 앱을 다시 시작해주세요.')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('추가 정보 입력'),
        automaticallyImplyLeading: false,
      ),
      body: Center(
        child: Card(
          margin: const EdgeInsets.all(20),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    '연구 진행과 RPP 크레딧 지급을 위해 추가 정보를 입력해주세요.',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 25),

                  DropdownButtonFormField<String>(
                    value: _selectedGender,
                    hint: const Text('성별 선택'),
                    items: _genders.map((String gender) {
                      return DropdownMenuItem<String>(
                        value: gender,
                        child: Text(gender),
                      );
                    }).toList(),
                    onChanged: (newValue) {
                      setState(() {
                        _selectedGender = newValue;
                      });
                    },
                    validator: (value) => value == null ? '성별을 선택해주세요.' : null,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '성별',
                    ),
                  ),
                  const SizedBox(height: 20),

                  DropdownButtonFormField<String>(
                    value: _selectedEducation,
                    hint: const Text('최종 학력 선택'),
                    isExpanded: true,
                    items: _educationLevels.map((String level) {
                      return DropdownMenuItem<String>(
                        value: level,
                        child: Text(level, overflow: TextOverflow.ellipsis),
                      );
                    }).toList(),
                    onChanged: (newValue) {
                      setState(() {
                        _selectedEducation = newValue;
                      });
                    },
                    validator: (value) => value == null ? '최종 학력을 선택해주세요.' : null,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '최종 학력',
                    ),
                  ),
                  const SizedBox(height: 30),

                  DropdownButtonFormField<String>(
                    value: _courseName,
                    hint: const Text('교과목명 선택'),
                    isExpanded: true,
                    items: _courseNames.map((String course) {
                      return DropdownMenuItem<String>(
                        value: course,
                        child: Text(course, overflow: TextOverflow.ellipsis),
                      );
                    }).toList(),
                    onChanged: (newValue) {
                      setState(() {
                        _courseName = newValue;
                      });
                    },
                    validator: (value) => value == null ? '교과목명을 선택해주세요.' : null,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '교과목명',
                    ),
                  ),
                  const SizedBox(height: 20),

                  DropdownButtonFormField<String>(
                    value: _classSection,
                    hint: const Text('분반 선택'),
                    items: _classSections.map((String section) {
                      return DropdownMenuItem<String>(
                        value: section,
                        child: Text(section),
                      );
                    }).toList(),
                    onChanged: (newValue) {
                      setState(() {
                        _classSection = newValue;
                      });
                    },
                    validator: (value) => value == null ? '분반을 선택해주세요.' : null,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '분반',
                    ),
                  ),
                  const SizedBox(height: 20),

                  DropdownButtonFormField<String>(
                    value: _professorName,
                    hint: const Text('교수명 선택'),
                    isExpanded: true,
                    items: _professorNames.map((String name) {
                      return DropdownMenuItem<String>(
                        value: name,
                        child: Text(name, overflow: TextOverflow.ellipsis),
                      );
                    }).toList(),
                    onChanged: (newValue) {
                      setState(() {
                        _professorName = newValue;
                      });
                    },
                    validator: (value) => value == null ? '교수명을 선택해주세요.' : null,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '교수명',
                    ),
                  ),
                  const SizedBox(height: 20),

                  TextFormField(
                    controller: _otherNotesController,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: '기타 특이사항 (선택 사항)',
                      hintText: '연구 결과에 영향을 미칠 수 있는 개인적인 특이사항이 있다면 자유롭게 기재해주세요. 예: 현재 복용 중인 정신과 약물이 있으신가요? , 기타 인지 기능에 영향을 줄 수 있는 상태가 있으신가요? 등',
                    ),
                    maxLines: 3,
                  ),
                  const SizedBox(height: 20),

                  if (_errorMessage != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 15),
                      child: Text(
                        _errorMessage!,
                        style: const TextStyle(color: Colors.red),
                        textAlign: TextAlign.center,
                      ),
                    ),

                  if (_isLoading)
                    const CircularProgressIndicator()
                  else
                    ElevatedButton(
                      onPressed: () => _submitAdditionalInfo(userUid),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size.fromHeight(50),
                        padding: const EdgeInsets.symmetric(vertical: 15)
                      ),
                      child: const Text('정보 제출 및 계속하기', style: TextStyle(fontSize: 16)),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}