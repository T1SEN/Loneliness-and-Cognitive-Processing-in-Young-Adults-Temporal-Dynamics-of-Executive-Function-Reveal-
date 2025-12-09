import 'dart:async';
import 'package:flutter/material.dart';
import 'dart:ui'; 

class RestPage extends StatefulWidget {
  final Duration duration;
  const RestPage({Key? key, required this.duration}) : super(key: key);

  @override
  State<RestPage> createState() => _RestPageState();
}

class _RestPageState extends State<RestPage> {
  late int _secondsLeft;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _secondsLeft = widget.duration.inSeconds;
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_secondsLeft == 0) {
        timer.cancel();
        if (mounted) {
          Navigator.pop(context, true); 
        }
      } else {
        if (mounted) {
          setState(() => _secondsLeft--);
        }
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final minutes = (_secondsLeft / 60).floor();
    final seconds = _secondsLeft % 60;
    
    return WillPopScope(
      onWillPop: () async => false, 
      child: Scaffold(
        backgroundColor: Colors.deepPurple.shade50,
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(32.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.local_cafe,
                  size: 80,
                  color: Colors.deepPurple.shade600,
                ),
                const SizedBox(height: 30),
                const Text(
                  "휴식 시간입니다",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 15),
                const Text(
                  "잠시 휴식을 취하세요.\n스트레칭이나 심호흡을 하시면 좋습니다.",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 18,
                    height: 1.5,
                    color: Colors.black54,
                  ),
                ),
                const SizedBox(height: 40),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Text(
                    "$minutes:${seconds.toString().padLeft(2, '0')}",
                    style: TextStyle(
                      fontSize: 56,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 2,
                      color: Colors.deepPurple.shade700,
                      fontFeatures: const [FontFeature.tabularFigures()],
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.orange.shade200),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.info_outline,
                        color: Colors.orange.shade700,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        "시간이 지나면 자동으로 다음 검사가 시작됩니다",
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.orange.shade700,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
} 