import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;

  Stream<User?> get authStateChanges => _auth.authStateChanges();

  User? get currentUser => _auth.currentUser;

  bool get isLoggedIn => currentUser != null;

  String? get studentId {
    final user = currentUser;
    if (user != null && user.email != null) {
      return user.email!.split('@')[0];
    }
    return null;
  }

  Future<void> signOut() async {
    await _auth.signOut();
  }
}

class AuthWrapper extends StatelessWidget {
  final Widget authenticatedRoute;
  final Widget unauthenticatedRoute;

  const AuthWrapper({
    Key? key,
    required this.authenticatedRoute,
    required this.unauthenticatedRoute,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.active) {
          if (snapshot.hasData) {
            return authenticatedRoute;
          }
          return unauthenticatedRoute;
        }

        return const Scaffold(
          body: Center(
            child: CircularProgressIndicator(),
          ),
        );
      },
    );
  }
}