import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  Map<String, dynamic> result = {};
  late TextEditingController controller;

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    controller = TextEditingController();
  }

  Future<String> predict(String sentence) async {
    try {
      String requestBody = json.encode({"sentence": sentence});

      http.Response response = await http.post(
        Uri.parse('http://127.0.0.1:5000/classify_spam'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: requestBody,
      );

      setState(() {
        result = jsonDecode(response.body);
      });
      if (response.statusCode == 200) {
        return response.body;
      } else {
        return 'Error: ${response.statusCode}';
      }
    } catch (e) {
      print('Error: $e');
      return 'Error: $e';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Spam Classifier'),
      ),
      body: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              style: const TextStyle(fontSize: 14),
              controller: controller,
              minLines: 3,
              maxLines: 5,
              decoration: InputDecoration(
                hintText: 'Enter the sentence',
                hintStyle: const TextStyle(fontSize: 14),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(color: Colors.blueGrey),
                ),
              ),
            ),
            const SizedBox(height: 40),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueGrey,
              ),
              onPressed: () {
                predict(controller.text);
              },
              child: const Text(
                'Predict',
                style: TextStyle(fontSize: 14, color: Colors.white),
              ),
            ),
            controller.text.isNotEmpty
                ? Column(
                    children: [
                      const SizedBox(
                        height: 40,
                        child: Divider(),
                      ),
                      const Text(
                        'OUTPUT',
                        style: TextStyle(fontSize: 14),
                      ),
                      const SizedBox(height: 30),
                      //output
                      Text.rich(
                        textAlign: TextAlign.center,
                        TextSpan(
                            text: controller.text,
                            style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.blue),
                            children: [
                              const TextSpan(
                                text: '\nis a\t',
                                style: TextStyle(
                                    fontSize: 16, color: Colors.black),
                              ),
                              TextSpan(
                                text: '${result['prediction']}\t',
                                style: TextStyle(
                                    fontSize: 16, color: Colors.green),
                              ),
                            ]),
                      ),
                    ],
                  )
                : const SizedBox()
          ],
        ),
      ),
    );
  }
}
