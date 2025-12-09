import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'pages/survey/ucla_page.dart';
import 'pages/survey/dass_page.dart';
import 'pages/ncft/stroop_page.dart';
import 'pages/ncft/wcst_page.dart';
import 'pages/ncft/prp_page.dart';
import 'pages/intro/intro_page.dart';
import 'pages/intro/ncft_intro_page.dart';
import 'pages/auth/auth_page.dart';
import 'pages/auth/additional_info_page.dart';
import 'services/auth_service.dart';
import 'pages/intro/end_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyResearchApp());
}

class MyResearchApp extends StatelessWidget {
  const MyResearchApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '전남대학교 신경심리랩 연구참여',
      theme: ThemeData(primarySwatch: Colors.blue),
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (context) => const IntroPage(),
        '/auth': (context) => const AuthPage(), 
        '/additional_info': (context) => const AdditionalInfoPage(),
        '/ucla': (context) => const UCLAPage(),
        '/dass': (context) => const DASS21Page(),
        '/cognitive_test_intro': (context) => const NCFTIntroPage(),
        '/stroop': (context) => const StroopPage(),
        '/wcst': (context) => const WcstPage(),
        '/prp': (context) => const PrpPage(),
        '/end_page': (context) => const EndPage(),
      },
    );
  }
}
