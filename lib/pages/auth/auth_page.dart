import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../../services/auth_service.dart';

class AuthPage extends StatefulWidget {
  const AuthPage({Key? key}) : super(key: key);

  @override
  _AuthPageState createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  final _formKey = GlobalKey<FormState>();
  final _studentIdController = TextEditingController();
  final _birthDateController = TextEditingController();
  bool _isLoading = false;
  bool _isLogin = true;
  String? _errorMessage;
  final AuthService _authService = AuthService();

  @override
  void dispose() {
    _studentIdController.dispose();
    _birthDateController.dispose();
    super.dispose();
  }

  String _createEmail(String studentId) {
    return '$studentId@id-login.univ';
  }

  int _calculateAge(String birthDateString) {
    if (birthDateString.length != 8) {
      // Consider how to handle this error in the UI, perhaps by returning null or a specific error code.
      // For now, throwing an error, but this should be handled gracefully in _submit or validator.
      throw ArgumentError('Birth date must be in YYYYMMDD format');
    }
    final year = int.tryParse(birthDateString.substring(0, 4));
    final month = int.tryParse(birthDateString.substring(4, 6));
    final day = int.tryParse(birthDateString.substring(6, 8));

    if (year == null || month == null || day == null) {
      throw ArgumentError('Invalid date components in birth date');
    }

    final birthDate = DateTime(year, month, day);
    final today = DateTime.now();
    int age = today.year - birthDate.year;
    if (today.month < birthDate.month ||
        (today.month == birthDate.month && today.day < birthDate.day)) {
      age--;
    }
    return age;
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final studentId = _studentIdController.text.trim();
    final birthDate = _birthDateController.text.trim();
    final email = _createEmail(studentId);

    try {
      UserCredential userCredential;
      if (_isLogin) {
        userCredential = await FirebaseAuth.instance.signInWithEmailAndPassword(
          email: email,
          password: birthDate,
        );
      } else {
        userCredential = await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: email,
          password: birthDate,
        );
        await FirebaseFirestore.instance.collection('participants').doc(userCredential.user!.uid).set({
          'studentId': studentId,
          'birthDate': birthDate,
          'age': _calculateAge(birthDate),
          'createdAt': Timestamp.now(),
        });
      }

      if (mounted) {
        final userUid = userCredential.user?.uid;
        if (userUid != null) {
            if (_isLogin) {
              Navigator.of(context).pushReplacementNamed('/ucla', arguments: userUid);
            } else {
              Navigator.of(context).pushReplacementNamed('/additional_info', arguments: userUid);
            }
        } else {
           setState(() {
             _errorMessage = '사용자 ID를 가져올 수 없습니다.';
           });
        }
      }
    } on FirebaseAuthException catch (e) {
      String message;
      if (_isLogin) {
        if (e.code == 'user-not-found' || e.code == 'invalid-credential' || e.code == 'wrong-password') {
           message = '학번 또는 생년월일이 일치하지 않습니다.';
        } else if (e.code == 'invalid-email') {
           message = '학번 형식이 올바르지 않습니다.';
        } else {
           message = '로그인 오류가 발생했습니다: ${e.message ?? e.code}';
        }
      } else {
        if (e.code == 'email-already-in-use') {
          message = '이미 가입된 학번입니다.';
        } else if (e.code == 'weak-password') {
          message = '생년월일(비밀번호)은 6자리 이상이어야 합니다.';
        } else if (e.code == 'invalid-email') {
           message = '학번 형식이 올바르지 않습니다.';
        } else {
          message = '회원가입 오류가 발생했습니다: ${e.message ?? e.code}';
        }
      }
      setState(() {
        _errorMessage = message;
      });
    } catch (e) {
      setState(() {
        _errorMessage = '예상치 못한 오류가 발생했습니다: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isAuthenticated = _authService.isLoggedIn;
    final studentId = _authService.studentId;

    return Scaffold(
      appBar: AppBar(
        title: Text(_isLogin ? '로그인' : '회원가입'),
        automaticallyImplyLeading: false,
      ),
      body: Center(
        child: isAuthenticated
            ? _buildAuthenticatedView(studentId)
            : _buildAuthForm(),
      ),
    );
  }

  Widget _buildAuthenticatedView(String? studentId) {
    return Card(
      margin: const EdgeInsets.all(20),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.check_circle_outline,
              size: 80,
              color: Colors.green,
            ),
            const SizedBox(height: 20),
            Text(
              '로그인 완료',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 10),
            if (studentId != null)
              Text(
                '학번: $studentId',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            const SizedBox(height: 20),
            const Text(
              '이제 연구에 참여하실 수 있습니다.\n다음 버튼을 클릭하여 계속하세요.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: () {
                final userUid = _authService.currentUser?.uid;
                if (userUid != null) {
                   Navigator.of(context).pushReplacementNamed('/ucla', arguments: userUid);
                } else {
                   ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('오류: 사용자 정보를 찾을 수 없습니다. 다시 로그인해주세요.')),
                   );
                }
              },
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(50),
              ),
              child: const Text('연구 참여 계속하기', style: TextStyle(fontSize: 16)),
            ),
            const SizedBox(height: 10),
            TextButton(
              onPressed: () async {
                await _authService.signOut();
                setState(() {});
              },
              child: const Text('로그아웃'),
            ),
          ],
        ),
      ),
    );
  }


  Widget _buildAuthForm() {
    return Card(
      margin: const EdgeInsets.all(20),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                _isLogin
                  ? '학번과 생년월일로 로그인하세요. 이 연구가 처음이시라면 회원가입 먼저 부탁드립니다'
                  : '처음 오셨다면?\n학번과 생년월일 8자리로 가입하세요.',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _studentIdController,
                decoration: const InputDecoration(labelText: '학번'),
                keyboardType: TextInputType.text,
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return '학번을 입력해주세요.';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _birthDateController,
                decoration: const InputDecoration(
                  labelText: '생년월일 8자리',
                  hintText: 'YYYYMMDD',
                ),
                keyboardType: TextInputType.number,
                obscureText: true,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '생년월일을 입력해주세요.';
                  }
                  if (value.length != 8 || int.tryParse(value) == null) {
                    return '생년월일은 8자리 숫자로 입력해주세요 (YYYYMMDD)';
                  }
                  if (value.length < 6) {
                    return '생년월일은 6자리 이상이어야 합니다.';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 20),
              if (_errorMessage != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.red),
                    textAlign: TextAlign.center,
                  ),
                ),
              if (_isLoading)
                const CircularProgressIndicator()
              else
                Column(
                  children: [
                    ElevatedButton(
                      onPressed: _submit,
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size.fromHeight(40),
                      ),
                      child: Text(_isLogin ? '로그인' : '회원가입'),
                    ),
                    TextButton(
                      onPressed: () {
                        setState(() {
                          _isLogin = !_isLogin;
                          _errorMessage = null;
                        });
                      },
                      child: Text(_isLogin ? '이 연구가 처음이시라면? 회원가입' : '이미 계정이 있으신가요? 로그인'),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }
}