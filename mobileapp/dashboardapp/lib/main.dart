import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_reactive_ble/flutter_reactive_ble.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DoseMate BLE Gateway',
      home: const BLEReceiverPage(),
    );
  }
}

class BLEReceiverPage extends StatefulWidget {
  const BLEReceiverPage({super.key});
  @override
  State<BLEReceiverPage> createState() => _BLEReceiverPageState();
}

class _BLEReceiverPageState extends State<BLEReceiverPage> {
  final flutterReactiveBle = FlutterReactiveBle();
  final serviceUuid = Uuid.parse("12345678-1234-1234-1234-1234567890ab");
  final characteristicUuid = Uuid.parse("abcd1234-5678-90ab-cdef-1234567890ab");

  DiscoveredDevice? device;
  Stream<List<int>>? bleStream;
  bool connected = false;
  String receivedJson = "";
  String predictionResult = "";

  @override
  void initState() {
    super.initState();
    startScan();
  }

  void startScan() {
    flutterReactiveBle.scanForDevices(
      withServices: [serviceUuid],
      scanMode: ScanMode.lowLatency,
    ).listen((device) {
      print("Found device: ${device.name} ${device.id}");
      if (device.name == "DoseMate-Test") {
        this.device = device;
        connectToDevice();
      }
    }, onError: (error) {
      print("BLE scan error: $error");
    });
  }

  void connectToDevice() {
    flutterReactiveBle.connectToDevice(id: device!.id).listen((connectionState) {
      if (connectionState.connectionState == DeviceConnectionState.connected) {
        print("Connected to ESP32 BLE");
        setState(() {
          connected = true;
        });
        subscribeToData();
      }
    }, onError: (error) {
      print("BLE connection error: $error");
    });
  }

  void subscribeToData() {
    final characteristic = QualifiedCharacteristic(
      serviceId: serviceUuid,
      characteristicId: characteristicUuid,
      deviceId: device!.id,
    );

    bleStream = flutterReactiveBle.subscribeToCharacteristic(characteristic);
    bleStream!.listen((data) {
      String jsonString = utf8.decode(data);
      print("Received JSON: $jsonString");

      setState(() {
        receivedJson = jsonString;
      });

      processAndSendToServer(jsonString);
    }, onError: (error) {
      print("BLE subscription error: $error");
    });
  }

  Future<void> processAndSendToServer(String jsonString) async {
    try {
      final Map<String, dynamic> parsed = jsonDecode(jsonString);

      // 🔧 Simulate dummy feature vector for now
      List<double> featureVector = [
        0.5, 0.3, 0.2, 0.8, 0.9, 0.4, 0.1, 0.7, 0.6, 0.3, 0.2
      ];

      // 🔧 Build ML server input
      final Map<String, dynamic> body = {
        "sequence": [featureVector]
      };

      final String formattedBody = jsonEncode(body);
      final serverUrl = Uri.parse("http://localhost:8000/predict_lstm");

      final response = await http.post(
        serverUrl,
        headers: {"Content-Type": "application/json"},
        body: formattedBody,
      );

      if (response.statusCode == 200) {
        final serverResponse = jsonDecode(response.body);
        print("Prediction received: ${serverResponse.toString()}");
        setState(() {
          predictionResult = serverResponse.toString();
        });
      } else {
        print("Server error: ${response.statusCode}");
      }

    } catch (e) {
      print("Error processing JSON: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("DoseMate BLE Gateway")),
      body: Center(
        child: connected
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text("Connected to ESP32!"),
                  const SizedBox(height: 20),
                  const Text("Received JSON:"),
                  const SizedBox(height: 10),
                  Text(receivedJson),
                  const SizedBox(height: 20),
                  const Text("Prediction:"),
                  const SizedBox(height: 10),
                  Text(predictionResult),
                ],
              )
            : const Text("Scanning for ESP32..."),
      ),
    );
  }
}