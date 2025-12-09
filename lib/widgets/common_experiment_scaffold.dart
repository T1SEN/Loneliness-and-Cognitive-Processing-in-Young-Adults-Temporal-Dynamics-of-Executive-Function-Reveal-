import 'package:flutter/material.dart';

class CommonExperimentScaffold extends StatelessWidget {
  final String title;
  final Widget body;
  final double? progressValue;
  final String? progressLabel;

  const CommonExperimentScaffold({
    Key? key,
    required this.title,
    required this.body,
    this.progressValue,
    this.progressLabel,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Column(
        children: [
          if (progressValue != null)
            LinearProgressIndicator(value: progressValue),
          if (progressLabel != null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(progressLabel!),
            ),
          Expanded(child: body),
        ],
      ),
    );
  }
}
