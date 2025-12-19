"""
비트코인 트레이딩 전략 - 딥러닝 모델 훈련 및 평가 스크립트

이 스크립트는 다음을 수행합니다:
1. 비트코인 가격 데이터 수집 및 전처리
2. Attention-Enhanced LSTM 모델 훈련
3. 확률 + RSI 기반 하이브리드 투자 전략 시뮬레이션
4. Buy and Hold 벤치마크와 비교
5. 결과 시각화 및 저장
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# utils.py에서 필요한 함수들 import
from utils import (
    load_bitcoin_data,
    create_features,
    prepare_data,
    evaluate_model,
    device
)

# 재현성을 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# 결과 저장 디렉토리 생성
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================
# 1. 모델 아키텍처 정의
# ============================================

class SelfAttention(nn.Module):
    """
    Self-Attention 레이어
    시퀀스 내의 중요한 시점에 더 많은 가중치를 부여합니다.
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context_vector, attention_weights


class MyTradingModel(nn.Module):
    """
    Attention-Enhanced LSTM 기반 비트코인 가격 방향 예측 모델
    
    아키텍처:
    - LSTM Layer 1: 시퀀스 패턴 학습
    - Self-Attention: 중요한 시점에 집중
    - LSTM Layer 2: 추상화된 특성 학습
    - Fully Connected Layers: 최종 분류
    
    출력: 상승 확률 (0~1)
    """
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super(MyTradingModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # First LSTM Layer
        self.lstm1 = nn.LSTM(
            input_size, hidden_size, 
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Self-Attention Layer
        self.attention = SelfAttention(hidden_size)
        
        # Second LSTM Layer
        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=1, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(dropout / 2)
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # First LSTM
        lstm_out, _ = self.lstm1(x)  # (batch, seq_len, hidden_size)
        lstm_out = self.dropout1(lstm_out)
        
        # BatchNorm (needs permutation)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, seq_len, hidden_size)
        
        # Self-Attention
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden_size)
        
        # Reshape for second LSTM
        context = context.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Second LSTM
        lstm_out, _ = self.lstm2(context)
        lstm_out = self.dropout2(lstm_out[:, -1, :])  # (batch, hidden_size//2)
        lstm_out = self.bn2(lstm_out)
        
        # Fully Connected Layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout4(out)
        
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out


# ============================================
# 2. 학습 함수
# ============================================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """
    모델 학습 함수 (Early Stopping 포함)
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"모델 학습 시작 (Device: {device})")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted.squeeze() == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted.squeeze() == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
                  f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ 학습 완료! Best Val Loss: {best_val_loss:.4f}")
    return history


def predict_with_probability(model, data_loader):
    """
    모델 예측 수행 및 확률 반환
    """
    model.eval()
    predictions_prob = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions_prob.append(outputs.cpu().numpy())
    
    predictions_prob = np.vstack(predictions_prob).flatten()
    predictions = (predictions_prob > 0.5).astype(int)
    
    return predictions_prob, predictions


# ============================================
# 3. 하이브리드 투자 전략
# ============================================

def simulate_hybrid_trading(predictions_prob, actual_prices, dates, rsi_values,
                            initial_capital=10000, transaction_fee=0.001,
                            threshold=0.6, position_scaling=True):
    """
    확률 + RSI 기반 하이브리드 트레이딩 전략
    
    전략 규칙:
    1. 상승 확률이 threshold 이상일 때만 매수
    2. 확률에 비례하여 포지션 크기 조절
    3. RSI 필터:
       - RSI > 70 (과매수): 투자 비율 50% 감소
       - RSI < 30 (과매도): 투자 비율 50% 증가
    4. 하락 예측 시 보유 자산 매도
    """
    cash = initial_capital
    btc_holdings = 0
    portfolio_values = []
    trade_log = []
    
    for i in range(len(predictions_prob)):
        current_price = actual_prices[i]
        prob = predictions_prob[i]
        rsi = rsi_values[i] if i < len(rsi_values) else 50  # RSI 기본값
        
        portfolio_value = cash + btc_holdings * current_price
        portfolio_values.append(portfolio_value)
        
        # 마지막 날 전량 매도
        if i == len(predictions_prob) - 1:
            if btc_holdings > 0:
                sell_value = btc_holdings * current_price * (1 - transaction_fee)
                trade_log.append({
                    'date': str(dates[i]),
                    'action': 'SELL_ALL',
                    'price': current_price,
                    'prob': prob,
                    'rsi': rsi,
                    'amount': btc_holdings,
                    'value': btc_holdings * current_price,
                    'fee': btc_holdings * current_price * transaction_fee
                })
                cash += sell_value
                btc_holdings = 0
            continue
        
        # 투자 비율 결정
        if position_scaling and prob > threshold:
            # 기본 투자 비율 = 확률
            invest_ratio = prob
            
            # RSI 필터 적용
            if rsi > 70:  # 과매수 상태
                invest_ratio *= 0.5  # 투자 비율 50% 감소
            elif rsi < 30:  # 과매도 상태
                invest_ratio = min(invest_ratio * 1.5, 1.0)  # 투자 비율 50% 증가
        elif prob > threshold:
            invest_ratio = 1.0
        else:
            invest_ratio = 0.0
        
        # 현재 포지션 비율
        current_btc_value = btc_holdings * current_price
        target_btc_value = portfolio_value * invest_ratio
        
        # 포지션 조정
        if target_btc_value > current_btc_value:  # 매수 필요
            buy_cash = min(target_btc_value - current_btc_value, cash)
            if buy_cash > 10:  # 최소 거래 금액
                buy_amount = (buy_cash * (1 - transaction_fee)) / current_price
                btc_holdings += buy_amount
                trade_log.append({
                    'date': str(dates[i]),
                    'action': 'BUY',
                    'price': current_price,
                    'prob': prob,
                    'rsi': rsi,
                    'amount': buy_amount,
                    'value': buy_cash,
                    'fee': buy_cash * transaction_fee
                })
                cash -= buy_cash
                
        elif target_btc_value < current_btc_value:  # 매도 필요
            sell_btc = min((current_btc_value - target_btc_value) / current_price, btc_holdings)
            if sell_btc * current_price > 10:  # 최소 거래 금액
                sell_value = sell_btc * current_price * (1 - transaction_fee)
                trade_log.append({
                    'date': str(dates[i]),
                    'action': 'SELL',
                    'price': current_price,
                    'prob': prob,
                    'rsi': rsi,
                    'amount': sell_btc,
                    'value': sell_btc * current_price,
                    'fee': sell_btc * current_price * transaction_fee
                })
                cash += sell_value
                btc_holdings -= sell_btc
    
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    total_trade_volume = sum(trade['value'] for trade in trade_log)
    total_fees_paid = sum(trade['fee'] for trade in trade_log)
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'portfolio_values': portfolio_values,
        'trade_log': trade_log,
        'num_trades': len(trade_log),
        'total_trade_volume': total_trade_volume,
        'total_fees_paid': total_fees_paid,
        'dates': dates
    }


# ============================================
# 4. 시각화 함수
# ============================================

def plot_training_history(history, save_path):
    """학습 과정 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2, color='#3498db')
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2, color='#e74c3c')
    axes[0].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2, color='#3498db')
    axes[1].plot(history["val_acc"], label="Validation Accuracy", linewidth=2, color='#e74c3c')
    axes[1].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 학습 그래프 저장: {save_path}")


def plot_portfolio_comparison(results_dict, dates, save_path):
    """포트폴리오 비교 시각화"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 포트폴리오 가치 변화
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for idx, (name, result) in enumerate(results_dict.items()):
        if 'portfolio_values' in result:
            style = '--' if name == 'Buy and Hold' else '-'
            lw = 2.5 if name == 'Buy and Hold' else 2
            axes[0].plot(
                dates[:len(result['portfolio_values'])], 
                result['portfolio_values'],
                label=f"{name} ({result['total_return']:.2f}%)",
                linewidth=lw, linestyle=style, color=colors[idx % len(colors)]
            )
    
    axes[0].axhline(y=10000, color='gray', linestyle=':', linewidth=1, label='Initial Capital')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 수익률 비교 바 차트
    strategies = list(results_dict.keys())
    returns = [results_dict[s]['total_return'] for s in strategies]
    bar_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
    
    bars = axes[1].bar(strategies, returns, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('Total Return Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return (%)')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.2f}%', ha='center', 
                    va='bottom' if ret > 0 else 'top', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 포트폴리오 비교 그래프 저장: {save_path}")


# ============================================
# 5. 메인 실행 함수
# ============================================

def create_sequences(X, y, seq_len=30):
    """시계열 시퀀스 데이터 생성"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("🚀 비트코인 트레이딩 전략 - 모델 훈련 및 평가")
    print("="*60)
    
    # ----- Step 1: 데이터 로딩 -----
    print("\n📥 Step 1: 데이터 로딩...")
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    btc_data = load_bitcoin_data(start_date=start_date, end_date=end_date)
    btc_features = create_features(btc_data, lookback_days=10)
    
    print(f"데이터 shape: {btc_features.shape}")
    print(f"기간: {btc_features.index[0]} ~ {btc_features.index[-1]}")
    
    # ----- Step 2: 데이터 분할 및 전처리 -----
    print("\n📊 Step 2: 데이터 분할 및 전처리...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        btc_features, test_size=0.2, validation_size=0.1
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 시퀀스 생성
    sequence_length = 30
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)
    
    print(f"시퀀스 데이터 shape: {X_train_seq.shape}")
    
    # DataLoader 생성
    batch_size = 32
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ----- Step 3: 모델 생성 및 학습 -----
    print("\n🧠 Step 3: 모델 생성 및 학습...")
    model = MyTradingModel(
        input_size=X_train_seq.shape[2],
        hidden_size=64,
        dropout=0.3
    ).to(device)
    
    print(f"모델 구조:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.001,
        patience=15
    )
    
    # 학습 그래프 저장
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_history.png"))
    
    # ----- Step 4: 모델 평가 -----
    print("\n📈 Step 4: 모델 평가...")
    my_prob, my_pred = predict_with_probability(model, test_loader)
    
    metrics = evaluate_model(y_test_seq, my_pred, model_name="MyTradingModel")
    
    print(f"예측 완료!")
    print(f"상승 예측: {np.sum(my_pred == 1)}개")
    print(f"하락 예측: {np.sum(my_pred == 0)}개")
    print(f"평균 상승 확률: {my_prob.mean():.2%}")
    
    # ----- Step 5: 트레이딩 시뮬레이션 -----
    print("\n💰 Step 5: 트레이딩 시뮬레이션...")
    
    # 테스트 데이터 준비
    test_start_idx = len(btc_features) - len(y_test) + sequence_length
    test_prices = btc_features["Close"].iloc[test_start_idx:test_start_idx+len(y_test_seq)].squeeze().values
    test_dates = btc_features.index[test_start_idx:test_start_idx+len(y_test_seq)]
    test_rsi = btc_features["RSI_14"].iloc[test_start_idx:test_start_idx+len(y_test_seq)].squeeze().values
    
    print(f"테스트 기간: {test_dates[0]} ~ {test_dates[-1]}")
    
    # 전략 1: 하이브리드 전략 (확률 + RSI)
    my_result = simulate_hybrid_trading(
        predictions_prob=my_prob,
        actual_prices=test_prices,
        dates=test_dates,
        rsi_values=test_rsi,
        initial_capital=10000,
        transaction_fee=0.001,
        threshold=0.6,
        position_scaling=True
    )
    
    # Buy and Hold 벤치마크
    initial_price = test_prices[0]
    coins_bought = (10000 * (1 - 0.001)) / initial_price
    buy_hold_final_value = coins_bought * test_prices[-1] * (1 - 0.001)
    buy_hold_return = (buy_hold_final_value - 10000) / 10000 * 100
    buy_hold_portfolio = [coins_bought * price for price in test_prices]
    
    buy_hold_result = {
        'initial_capital': 10000,
        'final_value': buy_hold_final_value,
        'total_return': buy_hold_return,
        'portfolio_values': buy_hold_portfolio,
        'num_trades': 2,
        'total_fees_paid': 10000 * 0.001 + coins_bought * test_prices[-1] * 0.001
    }
    
    # ----- Step 6: 결과 출력 및 저장 -----
    print("\n" + "="*70)
    print("📊 트레이딩 전략 결과 비교")
    print("="*70)
    
    print(f"\n{'전략':<25} {'최종자본':>15} {'수익률':>12} {'거래횟수':>10} {'수수료':>12}")
    print("-"*70)
    print(f"{'Buy and Hold':<25} ${buy_hold_result['final_value']:>14,.2f} {buy_hold_result['total_return']:>11.2f}% {buy_hold_result['num_trades']:>10} ${buy_hold_result['total_fees_paid']:>11,.2f}")
    print(f"{'My Hybrid Strategy':<25} ${my_result['final_value']:>14,.2f} {my_result['total_return']:>11.2f}% {my_result['num_trades']:>10} ${my_result['total_fees_paid']:>11,.2f}")
    print("-"*70)
    
    excess_return = my_result['total_return'] - buy_hold_result['total_return']
    print(f"\n📈 Buy and Hold 대비 초과 수익: {excess_return:+.2f}%p")
    
    if excess_return > 0:
        print("✅ 벤치마크를 초과했습니다!")
    else:
        print("❌ 벤치마크에 미달했습니다.")
    
    # 포트폴리오 비교 그래프 저장
    results_dict = {
        'Buy and Hold': buy_hold_result,
        'My Hybrid Strategy': my_result
    }
    plot_portfolio_comparison(results_dict, test_dates, os.path.join(RESULTS_DIR, "portfolio_comparison.png"))
    
    # 결과 JSON 저장
    results_json = {
        'test_period': {
            'start': str(test_dates[0]),
            'end': str(test_dates[-1])
        },
        'model_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        },
        'buy_and_hold': {
            'initial_capital': buy_hold_result['initial_capital'],
            'final_value': round(buy_hold_result['final_value'], 2),
            'total_return': round(buy_hold_result['total_return'], 2),
            'num_trades': buy_hold_result['num_trades']
        },
        'my_strategy': {
            'initial_capital': my_result['initial_capital'],
            'final_value': round(my_result['final_value'], 2),
            'total_return': round(my_result['total_return'], 2),
            'num_trades': my_result['num_trades'],
            'total_fees_paid': round(my_result['total_fees_paid'], 2)
        },
        'excess_return': round(excess_return, 2)
    }
    
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 결과 저장 완료: {RESULTS_DIR}/")
    print("  - training_history.png")
    print("  - portfolio_comparison.png")
    print("  - results.json")
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)
    
    return results_json


if __name__ == "__main__":
    results = main()
