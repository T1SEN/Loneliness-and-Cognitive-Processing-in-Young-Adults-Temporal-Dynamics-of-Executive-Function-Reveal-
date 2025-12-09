import 'package:flutter/material.dart';

class IntroPage extends StatelessWidget {
  const IntroPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    
    const String researchTitle = '젊은 성인의 외로움이 인지 기능(억제, 전환, 멀티태스킹)에 미치는 영향 연구';

    return Scaffold(
      appBar: AppBar(
        title: const Text('연구 참여 안내'),
        automaticallyImplyLeading: false, 
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '연구 참여 안내 및 동의서',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 20),

            const Text(
              '※ 귀하는 연구자가 수행하는 연구의 대상자로 적절하다고 판단되어 이 연구의 참여를 요청받게 되었습니다. 귀하가 이 연구에 참여할 것인지를 결정하기에 앞서 아래 설명문의 내용을 신중하게 읽어봐 주십시오. 그리고 궁금하신 사항은 언제든지 질문하셔도 됩니다. 이 연구는 귀하의 자발적인 참여로 수행될 것이므로, 이 연구와 관련 된 모든 내용을 이해하는 것이 중요합니다. 귀하께서 궁금해 하시는 모든 질문을 해 주시고 충분히 답변을 받았다고 생각될 때 이 동의서에 서명해 주십시오. 귀하의 서명 후 연구자 또한 이 서식에 자필로 서명하고 해당 날짜를 기재한 후 복사본을 귀하에게 전달할 것입니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),

            const Text(
              '연구 제목: $researchTitle',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('1. 연구의 목적'),
            const Text(
              '이 연구의 목적은 청년기 대학생들이 경험하는 주관적인 외로움 수준이, 일상생활과 학업 수행에 중요한 인지 능력, 특히 ① 집중력을 유지하고 방해 자극을 무시하는 능력(억제), ② 생각이나 행동을 유연하게 전환하는 능력(전환), 그리고 ③ 여러 가지 일을 동시에 처리할 때의 효율성(멀티태스킹/병목 처리)에 어떤 영향을 미치는지 알아보는 것입니다.\n\n'
              '또한, 외로움과 흔히 동반되는 우울, 불안, 스트레스와 같은 기분 상태의 영향을 제외하고도 외로움 자체만으로 이러한 인지 능력에 독립적인 영향을 주는지를 밝히고자 합니다. 이를 통해 청년기 외로움의 인지적 결과를 더 깊이 이해하고, 향후 외로움 관리 및 정신건강 증진 방안 마련에 기초 자료를 제공하고자 합니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('2. 연구대상자가 받게 될 각종 검사나 절차'),
            const Text(
              '귀하가 이 연구에 참여하시게 되면, 다음의 절차를 온라인 환경에서 귀하의 개인 컴퓨터(데스크톱 또는 노트북)를 이용하여 단일 세션으로 진행하게 됩니다.\n\n'
              '(사전 준비): 안정적인 인터넷 환경과 조용한 장소에서 개인 컴퓨터를 준비해 주십시오. 제공된 링크를 통해 연구용 홈페이지(Flutter 기반 웹 앱)에 접속하여 실행합니다. (모바일 기기 참여 불가)\n\n'
              '(연구 시작 및 동의): 프로그램을 실행하면, 본 설명문을 포함한 연구 안내를 확인하고 참여 여부에 대한 동의를 전자적으로(화면의 버튼 클릭 또는 체크박스 선택) 진행합니다.\n\n'
              '(설문 응답): 동의 후, 화면의 안내에 따라 귀하의 기본적인 인구통계 정보(나이, 성별 등), 외로움 수준(UCLA Loneliness Scale v.3), 그리고 최근의 기분 상태(우울, 불안, 스트레스 - DASS-21)에 관한 설문 문항에 응답합니다. (약 10-15분 소요 예상)\n\n'
              '(인지 과제 수행): 이어서 컴퓨터 화면에 제시되는 지시에 따라 세 가지 종류의 인지 과제를 수행합니다. 각 과제는 시작 전 충분한 설명과 연습 기회가 제공됩니다.\n'
              '  - Stroop 과제: 화면에 나타나는 단어의 \'색깔\'을 빠르고 정확하게 맞추는 과제입니다. (예: 파란색으로 쓰인 \'빨강\' 단어 → 파란색 키 누르기)\n'
              '  - WCST 과제: 화면에 제시되는 카드를 특정 규칙(색깔, 모양, 개수)에 따라 분류하는 과제입니다. 규칙은 계속 바뀌며, 피드백을 통해 새로운 규칙을 찾아 적용해야 합니다.\n'
              '  - PRP 과제: 짧은 시간 간격으로 두 가지 다른 과제(예: 화면 좌우 방향 판단 + 글자 종류 판단)를 순서대로, 최대한 빠르고 정확하게 수행하는 과제입니다.\n\n'
              '세 가지 인지 과제 수행에는 연습 및 과제 사이 휴식 시간을 포함하여 약 30-45분이 소요됩니다. 과제 순서는 참여자마다 무작위로 달라질 수 있습니다.\n\n'
              '(종료 및 보상 안내): 모든 설문과 과제가 완료되면 연구 참여가 종료되며, 보상 안내가 제공됩니다.\n\n'
              '총 예상 소요 시간: 약 40분 ~ 60분 입니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('3. 실험군 또는 대조군에 무작위 배정될 확률'),
            const Text(
              '이 연구는 특정 처치나 중재의 효과를 비교하는 연구가 아니므로, 실험군 또는 대조군 배정 절차는 해당 사항이 없습니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('4. 연구대상자가 준수해야 할 사항'),
            const Text(
              '이 연구에 정확하고 의미 있는 데이터를 얻기 위해, 귀하는 다음 사항들을 준수해 주시기를 부탁드립니다.\n\n'
              '- 연구 참여는 반드시 데스크톱 또는 랩탑 컴퓨터를 이용해 주십시오 (모바일 기기 불가).\n'
              '- 안정적인 인터넷 환경에서 참여해 주십시오 (유선 연결 권장).\n'
              '- 가능한 조용하고 방해받지 않는 환경에서 참여하여 과제에 집중해 주십시오.\n'
              '- 제시되는 모든 설문 문항에 솔직하고 성실하게 응답해 주십시오.\n'
              '- 각 인지 과제의 지시 사항을 잘 읽고 최대한 빠르고 정확하게 수행하도록 노력해 주십시오.\n'
              '- 연구 도중 임의로 프로그램을 종료하거나 다른 작업을 하지 말아 주십시오.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('5. 기대되는 이익'),
            const Text(
              '귀하가 이 연구에 직접적으로 참여함으로써 얻게 되는 의학적 또는 금전적 이익은 없습니다. 다만, 귀하의 참여를 통해 얻어진 연구 결과는 청년기 외로움과 인지 기능 간의 관계를 이해하는 데 중요한 학문적 기여를 할 수 있으며, 장기적으로는 외로움을 겪는 청년들을 위한 효과적인 지원 및 개입 전략 개발에 기초 자료로 활용될 수 있을 것으로 기대합니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

             _buildSectionTitle('6. 연구대상자에게 미칠 것으로 예견되는 위험이나 불편'),
            const Text(
              '본 연구는 비침습적인 온라인 설문 및 인지 과제로 구성되어 있어, 참여로 인한 위험은 최소한의 위험(Minimal Risk) 수준을 넘지 않을 것으로 예상됩니다. 그러나 다음과 같은 잠재적 불편감이나 위험이 있을 수 있습니다.\n\n'
              '- 정신적/심리적 불편감: 외로움, 우울, 불안, 스트레스 관련 설문에 응답하면서 일시적으로 불편하거나 부정적인 감정을 느낄 수 있습니다. 인지 과제를 수행하면서 집중력 요구, 실수 등으로 인해 약간의 좌절감, 지루함, 피로감, 또는 가벼운 수행 불안을 경험할 수 있습니다.\n'
              '- 신체적 불편감: 약 40-60분간 컴퓨터 화면을 보며 과제를 수행하므로, 경미한 눈의 피로나 어깨/목의 결림 등 일시적인 신체적 피로감을 느낄 수 있습니다.\n'
              '- 기술적 문제: 개인의 컴퓨터 환경이나 인터넷 연결 상태에 따라 프로그램 실행 오류, 중단, 데이터 전송 실패 등의 기술적인 문제가 발생할 수 있습니다.\n'
              '- 개인정보 유출 위험 (이론적): 귀하의 연구 데이터는 익명화되어 관리되지만, 온라인 데이터 처리 과정에서 이론적으로 정보 유출의 위험이 전혀 없다고는 할 수 없습니다. 연구자는 이를 방지하기 위해 12번 항목에 기술된 바와 같이 최선의 보안 조치를 취할 것입니다.\n\n'
              '만약 연구 참여 중 심리적 불편감이 심하게 느껴지거나 지속될 경우, 언제든지 참여를 중단하실 수 있으며, 필요한 경우 13번 항목에 안내된 기관의 도움을 받으실 수 있습니다. 기술적 문제 발생 시 연구 담당자에게 연락 주시면 가능한 해결 방안을 안내해 드리겠습니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('7. 연구 참여와 관련된 손상 발생 시 연구대상자에게 주어질 보상이나 치료방법'),
            const Text(
              '본 연구는 신체적 침습 행위가 없으므로 연구 참여로 인해 직접적인 신체적 손상이 발생할 가능성은 극히 낮습니다. 만일 예상치 못하게 본 연구 참여와 직접적으로 관련된 건강상의 문제(예: 극심한 정신적 스트레스 반응 등)가 발생하였다고 판단될 경우, 즉시 연구책임자에게 연락 주시기 바랍니다. 연구책임자는 상황을 파악하고 귀하가 적절한 상담이나 지원을 받을 수 있도록 관련 정보(예: 교내 상담센터)를 안내하는 등 최선의 조치를 취하겠습니다. (본 연구는 별도의 피해자 보상 보험에 가입되어 있지 않습니다.)',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('8. 연구 참여로 인해 받게 될 금전적 보상'),
            const Text(
              '귀하가 본 연구의 모든 절차(설문 응답 및 3가지 인지 과제 완료)를 완료하시면 감사의 의미로 심리학과 연구 참여 크레딧 6점을 부여해 드립니다. 크레딧은 연구 참여 확인 후, 절차에 따라 부여될 예정입니다.\n\n'
              '만약 연구 참여 도중 참여를 중단하시는 경우, 전체 예상 소요 시간(약 60분)의 50% 이상(약 30분 이상)을 성실히 수행하신 후 중단 시에는 심리학과 연구 참여 크레딧 3점을 부여해 드립니다. 전체 시간의 50% 미만을 수행하고 중단하시는 경우에는 별도의 보상이 지급되지 않습니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('9. 연구 참여로 인해 연구대상자가 부담해야 할 예상 비용'),
            const Text(
              '이 연구에 참여함으로써 귀하에게 직접적으로 요구되는 비용은 없습니다. 다만, 연구 참여에 소요되는 귀하의 시간과 개인 컴퓨터 및 인터넷 사용에 따른 간접적인 비용(예: 전기세, 인터넷 데이터 사용료 등)은 귀하께서 부담하시게 됩니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('10. 연구대상자가 선택할 수 있는 다른 중재'),
            const Text(
              '본 연구는 특정 질병의 치료나 증상 개선을 위한 중재 연구가 아닙니다. 따라서 귀하가 이 연구에 참여하지 않기로 결정하시더라도, 귀하가 현재 받고 있는 다른 치료나 일상생활에는 아무런 영향이 없습니다. 이 연구 참여 외에 다른 대안적인 선택지는 단순히 이 연구에 참여하지 않는 것입니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('11. 연구 참여 결정은 자발적인 것이며, 연구도중 언제라도 중도에 참여를 포기할 수 있음'),
            const Text(
              '귀하의 이 연구 참여 결정은 전적으로 자발적인 의사에 달려 있습니다. 귀하는 연구 참여를 거부할 수 있으며, 참여를 거부하시더라도 귀하에게는 어떠한 불이익도 없습니다. 또한, 연구 참여에 동의하신 후라도 연구가 진행되는 어느 시점에서든 이유를 밝히지 않고 자유롭게 참여를 중단하실 수 있습니다. 참여 중단 결정 역시 귀하에게 어떠한 불이익도 주지 않을 것입니다. 중단 시 보상에 대해서는 8번 항목을 참조해 주십시오.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

             _buildSectionTitle('12. 개인정보가 보장되지만, 연구자를 포함하여 관련자에게 자료가 보여 질 수 있음'),
            const Text(
              '귀하의 개인 정보 보호를 위해 최선을 다할 것입니다. 귀하의 이름이나 연락처 등 직접적인 개인 식별 정보는 연구 데이터와 함께 저장되지 않으며, 연구 데이터는 무작위로 생성된 고유 식별 코드로만 관리됩니다. 수집된 익명화된 데이터는 Google Cloud Platform의 서울 리전(asia-northeast3)에 위치한 Firebase 데이터베이스에 안전하게 저장 및 관리됩니다.\n\n'
              '수집된 익명화된 연구 데이터는 연구 목적(데이터 분석, 결과 보고, 논문 작성 등)으로만 사용됩니다. 귀하의 신원을 파악할 수 있는 기록은 기밀로 유지되며 공개적으로 열람되지 않습니다. 다만, 관련 법률이나 규정(예: "개인정보 보호법", "생명윤리 및 안전에 관한 법률")에 따라 연구의 신뢰성 검증 및 윤리적 수행 감독을 위해 전남대학교 생명윤리심의위원회(IRB), 연구 책임자의 소속 기관 관계자, 또는 필요한 경우 연구비 지원기관 등이 귀하의 익명화된 연구 자료를 직접 열람할 수 있습니다. 이러한 경우에도 귀하의 개인 정보는 최대한 보호될 것입니다. 귀하께서 이 설명문의 마지막에 동의를 표하시는 것은 이러한 제한적인 자료 열람 가능성에 동의함을 의미합니다.\n\n'
              '연구 결과가 학회나 학술지 등을 통해 발표될 경우, 오직 개인을 식별할 수 없는 형태로 통계 처리된 집단 수준의 결과만이 공개될 것입니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

             _buildSectionTitle('13. 연구와 관련한 새로운 정보가 수집되면 연구대상자에게 알려줌'),
            const Text(
              '연구 진행 중, 귀하의 연구 참여 지속 결정에 영향을 미칠 수 있는 중요한 새로운 정보(예: 연구의 위험이나 이익에 대한 중대한 변경사항)가 확인될 경우, 연구자는 이를 신속하게 귀하에게 알려드릴 것입니다.',
              
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('14. 심의위원회 연락처'),
            const Text(
              '이 연구는 전남대학교 생명윤리심의위원회(IRB)의 심의 및 승인을 받아 수행됩니다. 이 연구의 참여자로서 귀하의 권리에 대해 질문이 있거나 연구와 관련하여 불만 사항이 있으시면 전남대학교 생명윤리심의위원회(전화: 062-530-5932, 이메일: irb@jnu.ac.kr)에게 문의하실 수 있습니다(익명으로도 가능합니다).',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('15. 연구 참여를 제한하는 경우 및 해당 사유'),
            const Text(
              '다음과 같은 경우, 귀하의 동의와 관계없이 연구 참여가 중단되거나 귀하의 데이터가 최종 분석에서 제외될 수 있습니다.\n\n'
              '- 귀하가 연구 지시 사항을 따르지 않거나 불성실하게 연구에 참여한다고 판단될 경우 (예: 인지 과제 수행 정확도가 현저히 낮거나 반응 패턴이 비정상적인 경우 - 7번 항목 ③ 제외 기준 참조)\n'
              '- 연구 진행 중 기술적인 문제(예: 심각한 앱 오류, 지속적인 인터넷 연결 실패 등)로 인해 연구를 정상적으로 완료하기 어렵다고 판단될 경우\n'
              '- 연구 참여 도중 귀하가 이 연구의 선정/제외 기준에 부합하지 않는다는 사실이 확인될 경우 (예: 연구 중 색약임을 인지한 경우 등)',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),

            _buildSectionTitle('16. 연구 참여기간과 대략의 연구대상자 수'),
            const Text(
              '귀하가 이 연구에 참여하시는 기간은 단일 세션으로 약 40분 ~ 60분입니다. 이 연구에는 귀하를 포함하여 총 약 150명 ~ 200명의 대학생이 참여할 예정입니다.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            
            const Text(
              '연구대상자 동의서',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              '연구제목: $researchTitle',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              '1. 나는 연구자가 제공한 설명(문)을 통해 이 연구를 충분히 이해하였습니다.\n\n'
              '2. 나는 이 연구 참여로 인해 발생할 수 있는, 연구 참여에 대한 위험(불이익)과 이득(혜택)에 관하여 충분히 이해하였습니다.\n\n'
              '3. 나는 이 연구에 참여하는 것에 대하여 자발적으로 동의합니다.\n\n'
              '4. 나는 언제든지 연구의 참여를 거부하거나 연구의 참여를 중도에 철회할 수 있고 이러한 결정이 나에게 어떠한 해가 되지 않을 것이라는 것을 알고 있습니다.\n\n'
              '5. 나는 이 연구에 제공한 개인정보를 현행 법률과 생명윤리심의위원회 규정이 허용하는 범위 내에서 연구자가 수집하고 처리하는데 동의합니다.\n\n'
              '6. 나는 이 동의서 사본을 받을 것을 알고 있습니다.\n\n'
              '위의 모든 내용을 확인하고 이해하였으며, 연구 참여에 동의합니다.', 
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            
            Center(
              child: ElevatedButton(
                onPressed: () {
                  
                  Navigator.pushReplacementNamed(context, '/auth');
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  backgroundColor: Theme.of(context).primaryColor, 
                  foregroundColor: Colors.white, 
                ),
                child: const Text(
                  '동의하고 계속하기',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            ),
             const SizedBox(height: 24), 
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}