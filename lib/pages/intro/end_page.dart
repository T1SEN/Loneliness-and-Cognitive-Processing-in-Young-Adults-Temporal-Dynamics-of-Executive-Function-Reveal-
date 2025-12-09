import 'package:flutter/material.dart';

class EndPage extends StatelessWidget {
  const EndPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("검사 완료"),
        automaticallyImplyLeading: false, // Prevents back button
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Icon(
              Icons.check_circle_outline_rounded,
              size: 80,
              color: Colors.green,
            ),
            SizedBox(height: 20),
            Text(
              "고생하셨습니다!^^",
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            Text(
              "모든 신경인지기능검사가 성공적으로 완료되었습니다.\n\n창을 닫고 종료하시면 됩니다.\n크레딧 지급이나 기타 문의사항이 있다면 연구자 정승환(010-3204-2790)에게 연락주세요.",
              style: TextStyle(fontSize: 18),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 30),
            // Optionally, add a button to close the app or return to a main screen
            // For now, we'll just display the message.
          ],
        ),
      ),
    );
  }
} 