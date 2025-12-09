import 'package:flutter/material.dart';

class CommonInfoScreen extends StatelessWidget {
  final String title;
  final String content;
  final String buttonText;
  final VoidCallback onNext;

  const CommonInfoScreen({
    Key? key,
    required this.title,
    required this.content,
    required this.onNext,
    this.buttonText = "다음으로",
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Text(title,
                style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            Text(content, style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: onNext,
              child: Text(buttonText),
            ),
          ],
        ),
      ),
    );
  }
}
